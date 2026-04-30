from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Iterable

from langchain_chroma import Chroma

from .ingestion_state import ProcessedRecordStore, is_file_unchanged


@dataclass(frozen=True)
class IngestStats:
    files_total: int
    files_processed: int
    files_skipped: int


def _get_id(text: str) -> str:
    """为文本生成MD5哈希ID，用于向量库中的文档标识。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def discover_files(path: str, extensions: Iterable[str] = (".pdf", ".md")) -> list[str]:
    """发现指定路径下所有符合扩展名的文件，支持文件或目录路径。"""
    exts = {e.lower() for e in extensions}
    if os.path.isfile(path):
        return [path] if os.path.splitext(path)[1].lower() in exts else []
    if os.path.isdir(path):
        files: list[str] = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        return sorted(set(files))
    return []


def _vector_db_has_source(vector_db: Chroma, source_file_name: str) -> bool:
    """检查向量数据库中是否已存在指定来源文件的记录。"""
    existing = vector_db.get(where={"source": source_file_name})
    return bool(existing and len(existing.get("ids", [])) > 0)


def should_process_file(
    file_path: str, processed_record: dict[str, float], vector_db: Chroma
) -> bool:
    """判断文件是否需要重新处理：文件已修改或向量库中不存在。"""
    if not is_file_unchanged(file_path, processed_record):
        return True

    file_name = os.path.basename(file_path)
    return not _vector_db_has_source(vector_db, file_name)


def ingest_file(
    *,
    file_path: str,
    vector_db: Chroma,
    load_fn: Callable[[str], str],
    split_fn: Callable[[str], list[str]],
    session_id: str | None = None,
) -> int:
    """加载单个文件、切分为文本块并写入向量数据库，返回块数量。"""
    file_name = os.path.basename(file_path)
    sid = session_id or "default"
    existing = vector_db.get(
        where={"$and": [{"source": file_name}, {"session_id": sid}]}, include=[]
    )
    if existing.get("ids"):
        vector_db.delete(ids=existing["ids"])

    markdown_content = load_fn(file_path)
    chunks = split_fn(markdown_content)

    ids = [_get_id(chunk) for chunk in chunks]
    metadatas = [
        {"source": file_name, "chunk_index": i, "session_id": sid}
        for i in range(len(chunks))
    ]
    vector_db.add_texts(chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def process_path(
    *,
    path: str,
    vector_db: Chroma,
    record_store: ProcessedRecordStore,
    load_fn: Callable[[str], str],
    split_fn: Callable[[str], list[str]],
) -> IngestStats:
    """扫描路径下所有文件，对需要处理的文件执行加载、切分和入库操作。"""
    processed_record = record_store.load()
    files = discover_files(path)
    if not files:
        raise FileNotFoundError(f"路径不存在或无可处理文件: {path}")

    processed = 0
    skipped = 0

    for file_path in files:
        if not should_process_file(file_path, processed_record, vector_db):
            skipped += 1
            continue

        ingest_file(
            file_path=file_path,
            vector_db=vector_db,
            load_fn=load_fn,
            split_fn=split_fn,
        )
        processed_record[file_path] = os.path.getmtime(file_path)
        processed += 1

    record_store.save(processed_record)
    return IngestStats(
        files_total=len(files), files_processed=processed, files_skipped=skipped
    )
