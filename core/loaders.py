from __future__ import annotations

import os

from .pdf_to_markdown import pdf_to_markdown


def load_document(file_path: str) -> str:
    """根据文件扩展名加载文档内容，支持PDF（转Markdown）和Markdown文件。"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return pdf_to_markdown(file_path)
    if ext == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError(f"不支持的文件类型: {ext}，仅支持 .pdf 或 .md")
