from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedRecordStore:
    path: str

    def load(self) -> dict[str, float]:
        """从JSON文件加载已处理文件记录（文件路径到修改时间的映射）。"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {k: float(v) for k, v in data.items()}
            except (json.JSONDecodeError, ValueError, OSError):
                print(f"[record] 解析失败，将重建: {self.path}")
                return {}
        return {}

    def save(self, record: dict[str, float]) -> None:
        """将已处理文件记录保存到JSON文件。"""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)


def is_file_unchanged(file_path: str, record: dict[str, float]) -> bool:
    """检查文件是否自上次处理以来未发生修改。"""
    if file_path not in record:
        return False
    current_mtime = os.path.getmtime(file_path)
    return float(record[file_path]) == float(current_mtime)
