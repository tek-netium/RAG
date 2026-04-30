#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估脚本 - 基于原有 main.py 代码修改，用于测试 embedding + BM25 + DeepSeek 的检索生成质量
使用方法：
    python evaluate.py
"""

import os
import hashlib
from datasets import Dataset

# 原有导入（与 main.py 一致的核心能力）
# 注意：这里不再使用顶层兼容层文件（load_document.py / text_splitter.py）
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

# RAGAS 评估导入
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config import SYSTEM_PROMPT as prompt

# ======================== 配置 ========================
# RAG 生成阶段：本地 Ollama（Qwen）
llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0.2,
    num_predict=16384,
)

# 评估阶段：保留 DeepSeek API（RAGAS 会用它来打分）
if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError(
        "请在系统中配置环境变量 DEEPSEEK_API_KEY（用于评估阶段 DeepSeek API）"
    )

eval_llm = ChatDeepSeek(
    model="deepseek-chat",  # 或换成 deepseek-chat（更稳的情况也不少）
    temperature=0.0,
    max_tokens=8192,  # 先拉高到 4096/8192 试
    timeout=300,
    max_retries=4,
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
)


def get_id(text: str) -> str:
    """为文本生成MD5哈希ID。"""
    return hashlib.md5(text.encode()).hexdigest()


# ======================== 加载已有数据库（不处理新文件） ========================
vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# 检查数据库是否为空
if not vector_db.get()["documents"]:
    raise RuntimeError("向量数据库为空，请先运行原始 main.py 导入文档。")


# ======================== 构建混合检索器（与 main.py 一致） ========================
def build_ensemble_retriever_from_db(vector_db, k=5, weights=(0.7, 0.3)):
    """从已有向量数据库中构建混合检索器（向量检索+BM25）。"""
    all_docs_in_db = vector_db.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=doc, metadata=meta if meta is not None else {})
        for doc, meta in zip(all_docs_in_db["documents"], all_docs_in_db["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights,
    )
    return ensemble_retriever


ensemble_retriever = build_ensemble_retriever_from_db(vector_db)


# ======================== 定义检索工具（与 main.py 完全一致） ========================
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant document chunks from the vector database for a given query."""
    unique_docs = ensemble_retriever.invoke(query)
    seen = set()
    deduped_docs = []
    for doc in unique_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped_docs.append(doc)
    serialized = "\n\n".join(
        f"Source file: {doc.metadata.get('source', 'unknown')}\n"
        f"Chunk index: {doc.metadata.get('chunk_index', 'N/A')}\n"
        f"Content: {doc.page_content}"
        for doc in deduped_docs
    )
    return serialized, deduped_docs


# ======================== 创建 Agent（prompt 与 main.py 一致） ========================

tools = [retrieve_context]
agent = create_agent(llm, tools, system_prompt=prompt)


# ======================== 辅助函数：从 agent 消息中提取上下文和答案 ========================
def extract_contexts_and_answer(agent, query: str):
    """
    流式调用Agent，从中提取检索到的上下文列表和最终答案。
    """
    contexts = []
    final_answer = ""

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 30},
        stream_mode="values",
    ):
        msg = event["messages"][-1]
        # 提取工具消息中的上下文（retrieve_context 的返回）
        if hasattr(msg, "type") and msg.type == "tool":
            # msg.content 是 serialized 字符串，格式如 "Source file: ...\nContent: ..."
            blocks = msg.content.split("\n\n")
            for block in blocks:
                if "Content:" in block:
                    content = block.split("Content:", 1)[-1].strip()
                    contexts.append(content)

        # 提取最终答案（AI 消息且无 tool_calls）
        if hasattr(msg, "type") and msg.type == "ai" and not msg.tool_calls:
            final_answer = msg.content

    # 如果没有找到 final_answer，尝试取最后一条非工具消息
    if not final_answer:
        # 简单处理：重新模拟一次取最后一条消息（此处简化，实际可按需完善）
        pass

    return contexts, final_answer


# ======================== 主评估流程 ========================
def main():
    """RAGAS评估主流程：对测试查询生成答案并使用RAGAS指标打分。"""
    print("=" * 60)
    print("评估模式：测试 embedding + BM25 混合检索 + qwen3.5:9B 生成质量")
    print("=" * 60)

    # 定义测试查询（请根据实际修改）
    test_queries = [
        "在实数结构内等式与不等式的证明思路是什么？",
    ]

    test_data = {
        "question": [],
        "answer": [],
        "contexts": [],
    }

    for idx, query in enumerate(test_queries, 1):
        print(f"\n处理查询 {idx}/{len(test_queries)}: {query}")
        contexts, answer = extract_contexts_and_answer(agent, query)
        print(f"  检索到 {len(contexts)} 个上下文片段")
        print(f"  生成答案: {answer[:150]}...")

        test_data["question"].append(query)
        test_data["answer"].append(answer)
        test_data["contexts"].append(contexts)

    # 使用 RAGAS 评估
    print("\n开始 RAGAS 评估...")
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # DeepSeek 目前仅支持 n=1；而 RAGAS 的 `answer_relevancy` 默认 strictness=3 会触发 n=3。
    # 显式降为 1，避免该指标因 400/超时而变成 nan。
    answer_relevancy.strictness = 1

    metrics = [
        faithfulness,  # 答案忠实度
        answer_relevancy,  # 答案相关性
    ]

    dataset = Dataset.from_dict(test_data)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n" + "=" * 60)
    print("评估结果（分数范围 0~1，越高越好）")
    print("=" * 60)
    print(result)

    print("\n评估完成。")


if __name__ == "__main__":
    main()
