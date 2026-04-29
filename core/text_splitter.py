from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def text_splitter(text: str) -> list[str]:
    """按段落和字符边界递归切分文本为重叠的块，每块约1000字符。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
        keep_separator=True,
        is_separator_regex=False,
    )
    return splitter.split_text(text)
