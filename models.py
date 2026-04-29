from __future__ import annotations

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from config import AppConfig


def build_llm(cfg: AppConfig):
    """根据配置构建Ollama本地大语言模型实例。"""
    return ChatOllama(
        model=cfg.ollama_model,
        temperature=cfg.ollama_temperature,
        num_predict=cfg.ollama_num_predict,
    )


def build_embeddings(cfg: AppConfig):
    """根据配置构建HuggingFace嵌入模型实例。"""
    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model_name,
        model_kwargs={"device": cfg.embedding_device},
        encode_kwargs={"normalize_embeddings": cfg.embedding_normalize},
    )
