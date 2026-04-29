from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config import AppConfig, SYSTEM_PROMPT
from models import build_embeddings, build_llm

from .agent import build_agent_with_memory, make_redis_history_factory
from .document_processor import process_path
from .ingestion_state import ProcessedRecordStore
from .loaders import load_document
from .retriever import build_ensemble_retriever, make_retrieve_context_tool
from .text_splitter import text_splitter
from .vectorstore import build_vector_db


@dataclass(frozen=True)
class RagApp:
    agent_with_memory: object
    clear_history: Callable[[], None]


def build_rag_app(
    path: str, cfg: AppConfig, *, session_id: str = "main_session"
) -> RagApp:
    """构建完整的RAG应用实例，包括向量库、混合检索器、Agent和对话历史。"""
    llm = build_llm(cfg)
    embeddings = build_embeddings(cfg)

    vector_db = build_vector_db(
        embedding_function=embeddings,
        persist_directory=cfg.chroma_persist_dir,
    )

    process_path(
        path=path,
        vector_db=vector_db,
        record_store=ProcessedRecordStore(cfg.processed_record_file),
        load_fn=load_document,
        split_fn=text_splitter,
    )

    ensemble = build_ensemble_retriever(
        vector_db=vector_db,
        vector_k=cfg.vector_k,
        bm25_k=cfg.bm25_k,
        weights=cfg.ensemble_weights,
    )
    tools = [make_retrieve_context_tool(ensemble)]

    history_factory = make_redis_history_factory(
        url=cfg.redis_url,
        key_prefix=cfg.redis_key_prefix,
        ttl_seconds=cfg.redis_ttl_seconds,
    )

    agent_with_memory = build_agent_with_memory(
        llm=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        get_session_history=history_factory,
    )

    def _clear_history() -> None:
        history_factory(session_id).clear()

    return RagApp(agent_with_memory=agent_with_memory, clear_history=_clear_history)
