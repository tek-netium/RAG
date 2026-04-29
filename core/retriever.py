from __future__ import annotations

from typing import Iterable, Sequence

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def build_vector_retriever(vector_db: Chroma, *, k: int):
    """基于Chroma向量数据库构建向量检索器，返回top-k结果。"""
    return vector_db.as_retriever(search_kwargs={"k": k})


def build_bm25_retriever_from_chroma(vector_db: Chroma, *, k: int) -> BM25Retriever:
    """从Chroma向量库中提取全部文档构建BM25关键词检索器。"""
    all_docs_in_db = vector_db.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=doc, metadata=meta if meta is not None else {})
        for doc, meta in zip(all_docs_in_db["documents"], all_docs_in_db["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    return bm25


def build_ensemble_retriever(
    *,
    vector_db: Chroma,
    vector_k: int,
    bm25_k: int,
    weights: Sequence[float] = (0.7, 0.3),
) -> EnsembleRetriever:
    """构建融合向量检索与BM25检索的混合检索器，按权重合并排序结果。"""
    vector_retriever = build_vector_retriever(vector_db, k=vector_k)
    bm25_retriever = build_bm25_retriever_from_chroma(vector_db, k=bm25_k)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=list(weights),
    )


def dedupe_docs_by_content(docs: Iterable[Document]) -> list[Document]:
    """按文档内容去重，返回不重复的文档列表。"""
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        out.append(doc)
    return out


def serialize_docs(docs: Iterable[Document]) -> str:
    """将文档列表序列化为带来源信息和来源汇总的格式化字符串。"""
    docs_list = list(docs)
    sources: list[str] = []
    seen_sources: set[str] = set()
    for doc in docs_list:
        src = str(doc.metadata.get("source", "unknown"))
        if src not in seen_sources:
            sources.append(src)
            seen_sources.add(src)

    parts: list[str] = []
    if sources:
        parts.append("SOURCES:\n" + "\n".join(f"- {s}" for s in sources))

    for doc in docs_list:
        parts.append(
            "Source file: "
            + str(doc.metadata.get("source", "unknown"))
            + "\nChunk index: "
            + str(doc.metadata.get("chunk_index", "N/A"))
            + "\nContent: "
            + doc.page_content
        )
    return "\n\n".join(parts)


def make_retrieve_context_tool(ensemble_retriever: EnsembleRetriever):
    """创建LangChain检索工具，供Agent调用以检索相关文档上下文。"""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant document chunks from the vector database for a given query."""
        docs = ensemble_retriever.invoke(query)
        deduped = dedupe_docs_by_content(docs)
        return serialize_docs(deduped), deduped

    return retrieve_context
