from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from config import DEFAULT_CONFIG, AppConfig, SYSTEM_PROMPT
from core.agent import (
    build_agent_with_memory,
    make_redis_history_factory,
    stream_events,
)
from core.document_processor import ingest_file, process_path
from core.ingestion_state import ProcessedRecordStore
from core.loaders import load_document
from core.retriever import build_ensemble_retriever, make_retrieve_context_tool
from core.text_splitter import text_splitter
from core.vectorstore import build_vector_db
from models import build_embeddings, build_llm


def _sse(data: Any) -> str:
    """将数据格式化为Server-Sent Events格式的字符串。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _initial_ingest_path(cfg: AppConfig) -> str:
    """获取启动时初始文档路径，优先使用环境变量，否则使用默认值。"""
    return os.environ.get("RAG_INITIAL_PATH") or "test.pdf"


def _upload_dir() -> str:
    """获取文件上传目录路径。"""
    return os.environ.get("RAG_UPLOAD_DIR") or "./uploads"


def _find_uploaded_file(filename: str) -> list[str]:
    """在上传目录中查找匹配文件名的所有路径（支持子目录）。"""
    upload_dir = os.path.abspath(_upload_dir())
    found: list[str] = []
    if not os.path.isdir(upload_dir):
        return found
    for root, _dirs, files in os.walk(upload_dir):
        for f in files:
            if f == filename:
                found.append(os.path.join(root, f))
    return found


def _build_retriever_and_tools(
    *, vector_db, cfg: AppConfig, session_id: str | None = None
):
    """构建检索器和工具列表，空数据库时降级为纯向量检索器。"""
    try:
        ensemble = build_ensemble_retriever(
            vector_db=vector_db,
            vector_k=cfg.vector_k,
            bm25_k=cfg.bm25_k,
            weights=cfg.ensemble_weights,
            session_id=session_id,
        )
        return ensemble, [make_retrieve_context_tool(ensemble)]
    except Exception:
        # Fallback: vector retriever only, wrapped into the same tool API shape.
        search_kwargs: dict = {"k": cfg.vector_k}
        if session_id:
            search_kwargs["filter"] = {"session_id": session_id}
        vector_retriever = vector_db.as_retriever(search_kwargs=search_kwargs)

        def _retrieve_context(query: str):
            docs = vector_retriever.invoke(query)
            # If there is no content, return empty context.
            if not docs:
                return "", []
            # Reuse the serializer from core.retriever to match prompt expectations.
            from core.retriever import dedupe_docs_by_content, serialize_docs

            deduped = dedupe_docs_by_content(docs)
            return serialize_docs(deduped), deduped

        # Match langchain tool signature via decorator if available, otherwise raw callable.
        try:
            from langchain.tools import tool  # type: ignore

            wrapped = tool(response_format="content_and_artifact")(_retrieve_context)
            return vector_retriever, [wrapped]
        except Exception:
            return vector_retriever, [_retrieve_context]


def _rebuild_agent(
    *, vector_db, cfg: AppConfig, history_factory, session_id: str | None = None
):
    """文档变更后热重建Agent、检索器和工具，无需重启服务。"""
    llm = build_llm(cfg)
    ensemble_or_retriever, tools = _build_retriever_and_tools(
        vector_db=vector_db, cfg=cfg, session_id=session_id
    )
    agent_with_memory = build_agent_with_memory(
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        get_session_history=history_factory,
    )
    return agent_with_memory, ensemble_or_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：初始化向量库、加载初始文档、构建Agent。"""
    cfg = DEFAULT_CONFIG
    initial_path = _initial_ingest_path(cfg)

    llm = build_llm(cfg)
    embeddings = build_embeddings(cfg)

    vector_db = build_vector_db(
        embedding_function=embeddings,
        persist_directory=cfg.chroma_persist_dir,
    )

    record_store = ProcessedRecordStore(cfg.processed_record_file)

    # Best-effort initial ingestion. If the path doesn't exist or has no supported
    # files, we start the service anyway so documents can be uploaded later.
    try:
        process_path(
            path=initial_path,
            vector_db=vector_db,
            record_store=record_store,
            load_fn=load_document,
            split_fn=text_splitter,
        )
    except Exception as e:
        print(f"[启动提示] 初始文档未加载：{e}")

    history_factory = make_redis_history_factory(
        url=cfg.redis_url,
        key_prefix=cfg.redis_key_prefix,
        ttl_seconds=cfg.redis_ttl_seconds,
    )

    app.state.cfg = cfg
    app.state.vector_db = vector_db
    app.state.record_store = record_store
    app.state.history_factory = history_factory
    app.state.session_agents: dict[str, object] = {}

    yield


def _get_redis():
    """获取Redis客户端实例，不可用时返回None。"""
    try:
        import redis as redis_client

        return redis_client.from_url(DEFAULT_CONFIG.redis_url)
    except Exception:
        return None


def _register_session(session_id: str):
    """将session_id注册到Redis会话列表中。"""
    r = _get_redis()
    if r is None:
        return
    try:
        r.sadd("rag:session_ids", session_id)
    except Exception:
        pass


def _get_or_create_session_agent(session_id: str):
    """获取或创建指定会话的Agent和检索器。"""
    if session_id not in app.state.session_agents:
        cfg: AppConfig = app.state.cfg
        agent, retriever = _rebuild_agent(
            vector_db=app.state.vector_db,
            cfg=cfg,
            history_factory=app.state.history_factory,
            session_id=session_id,
        )
        app.state.session_agents[session_id] = (agent, retriever)
    return app.state.session_agents[session_id]


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_web_session"


def _chat_sse_stream(*, session_id: str, user_message: str) -> Iterator[str]:
    """生成聊天SSE流式事件，包括回答增量、检索上下文和错误信息。"""
    _register_session(session_id)
    agent_with_memory = _get_or_create_session_agent(session_id)[0]
    try:
        for evt in stream_events(
            agent_with_memory=agent_with_memory,
            session_id=session_id,
            user_input=user_message,
        ):
            if evt["type"] == "answer_delta":
                yield _sse({"type": "answer_delta", "content": evt["text"]})
            elif evt["type"] == "retrieved_context":
                yield _sse({"type": "retrieved_context", "content": evt["text"]})
    except Exception as e:
        yield _sse({"type": "error", "content": str(e)})
    yield _sse({"type": "done"})


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """处理聊天请求，返回SSE流式响应。"""
    return StreamingResponse(
        iterate_in_threadpool(
            _chat_sse_stream(session_id=req.session_id, user_message=req.message)
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """获取指定会话的完整聊天历史。"""
    try:
        history = app.state.history_factory(session_id)
        return {
            "session_id": session_id,
            "history": [
                {"role": msg.type, "content": msg.content} for msg in history.messages
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """清除指定会话的聊天历史。"""
    try:
        app.state.history_factory(session_id).clear()
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    session_id: str = Query("default"),
):
    """上传文档并异步处理入库，自动热重建检索器。"""
    upload_dir = _upload_dir()
    file_path = os.path.join(upload_dir, file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    sid = session_id

    def process_one() -> None:
        try:
            cfg: AppConfig = app.state.cfg
            vector_db = app.state.vector_db

            ingest_file(
                file_path=file_path,
                vector_db=vector_db,
                load_fn=load_document,
                split_fn=text_splitter,
                session_id=sid,
            )

            # Update processed record so future ingestions can skip unchanged files.
            record_store: ProcessedRecordStore = app.state.record_store
            record = record_store.load()
            record[file_path] = float(os.path.getmtime(file_path))
            record_store.save(record)

            # Hot-rebuild session-specific agent if it exists
            if sid in app.state.session_agents:
                history_factory = app.state.history_factory
                agent_with_memory, retriever = _rebuild_agent(
                    vector_db=vector_db,
                    cfg=cfg,
                    history_factory=history_factory,
                    session_id=sid,
                )
                app.state.session_agents[sid] = (agent_with_memory, retriever)
            print(f"[文档] 已处理并更新检索器 [{sid}]: {file.filename}")
        except Exception as e:
            print(f"[文档] 处理失败 [{sid}]: {file.filename} - {e}")

    if background_tasks is not None:
        background_tasks.add_task(process_one)
    else:
        process_one()

    return {"status": "processing", "filename": file.filename}


@app.get("/documents")
async def list_documents(session_id: str = Query("default")):
    vector_db = app.state.vector_db
    all_meta = vector_db.get(include=["metadatas"])
    sources: set[str] = set()
    for meta in all_meta.get("metadatas", []):
        if not meta or "source" not in meta:
            continue
        doc_sid = meta.get("session_id")
        if doc_sid == session_id or (doc_sid is None and session_id == "default"):
            sources.add(str(meta["source"]))
    return {"documents": sorted(sources)}


@app.delete("/documents/{filename}")
async def delete_document(filename: str, session_id: str = Query("default")):
    try:
        vector_db = app.state.vector_db
        # Try exact match first
        results = vector_db.get(
            where={"$and": [{"source": filename}, {"session_id": session_id}]},
            include=[],
        )
        if not results.get("ids") and session_id == "default":
            # Backward compat: old docs without session_id field
            results = vector_db.get(
                where={"source": filename},
                include=[],
            )
            if results.get("ids"):
                all_meta = vector_db.get(ids=results["ids"], include=["metadatas"])
                keep_ids = [
                    i
                    for i, m in zip(results["ids"], all_meta.get("metadatas", []))
                    if m
                    and (
                        m.get("session_id") == "default" or m.get("session_id") is None
                    )
                ]
                results = {"ids": keep_ids}
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            vector_db.delete(ids=ids_to_delete)

        # Also remove matching entries from processed record store
        record_store: ProcessedRecordStore = app.state.record_store
        record = record_store.load()
        new_record = {
            k: v for k, v in record.items() if os.path.basename(k) != filename
        }
        record_store.save(new_record)

        # Delete physical file from uploads
        found_files = _find_uploaded_file(filename)
        print(
            f"[文档] 查找物理文件 '{filename}': 找到 {len(found_files)} 个: {found_files}"
        )
        for file_path in found_files:
            try:
                os.remove(file_path)
                print(f"[文档] 已删除物理文件: {file_path}")
            except OSError as exc:
                print(f"[文档] 删除物理文件失败: {file_path} - {exc}")

        # Rebuild session agent if it exists (ignore build failures)
        if session_id in app.state.session_agents:
            try:
                cfg: AppConfig = app.state.cfg
                history_factory = app.state.history_factory
                agent_with_memory, retriever = _rebuild_agent(
                    vector_db=vector_db,
                    cfg=cfg,
                    history_factory=history_factory,
                    session_id=session_id,
                )
                app.state.session_agents[session_id] = (agent_with_memory, retriever)
            except Exception:
                app.state.session_agents.pop(session_id, None)
                print(f"[文档] 会话 {session_id} 中无更多文档，已移除 Agent 缓存")

        return {"status": "deleted", "filename": filename}
    except Exception as e:
        print(f"[文档] 删除失败 '{filename}': {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SessionCreate(BaseModel):
    name: str = ""


@app.post("/sessions")
async def create_session(req: SessionCreate):
    session_id = uuid.uuid4().hex[:12]
    name = req.name or f"会话 {session_id[:6]}"
    r = _get_redis()
    if r is not None:
        try:
            r.hset("rag:session_names", session_id, name)
            r.sadd("rag:session_ids", session_id)
        except Exception:
            pass
    return {"session_id": session_id, "name": name}


@app.get("/sessions")
async def list_sessions():
    r = _get_redis()
    if r is None:
        return {"sessions": []}
    try:
        session_ids = r.smembers("rag:session_ids")
        names = r.hgetall("rag:session_names")

        sessions = []
        for sid in session_ids:
            sid = sid.decode() if isinstance(sid, bytes) else sid
            name = names.get(sid.encode() if isinstance(sid, str) else sid)
            if isinstance(name, bytes):
                name = name.decode()
            sessions.append({"session_id": sid, "name": name or sid[:8]})

        sessions.sort(key=lambda s: s["session_id"])
        return {"sessions": sessions}
    except Exception:
        return {"sessions": []}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        # Clear chat history
        app.state.history_factory(session_id).clear()

        # Delete all documents for this session
        vector_db = app.state.vector_db
        results = vector_db.get(where={"session_id": session_id}, include=[])
        if results.get("ids"):
            vector_db.delete(ids=results["ids"])

        # Remove session agent
        app.state.session_agents.pop(session_id, None)

        # Remove from Redis
        r = _get_redis()
        if r is not None:
            try:
                r.srem("rag:session_ids", session_id)
                r.hdel("rag:session_names", session_id)
            except Exception:
                pass

        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
