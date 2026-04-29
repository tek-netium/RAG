from __future__ import annotations

from langchain_chroma import Chroma


def build_vector_db(*, embedding_function, persist_directory: str) -> Chroma:
    """构建并返回Chroma向量数据库实例，支持持久化存储。"""
    return Chroma(
        embedding_function=embedding_function, persist_directory=persist_directory
    )
