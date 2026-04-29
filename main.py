import sys

from config import DEFAULT_CONFIG
from core.agent import run_cli
from core.app_factory import build_rag_app


def main(path: str):
    """构建RAG应用并启动命令行交互会话。"""
    session_id = "main_session"
    app = build_rag_app(path, DEFAULT_CONFIG, session_id=session_id)
    run_cli(
        agent_with_memory=app.agent_with_memory,
        session_id=session_id,
        clear_history_fn=app.clear_history,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("test.pdf")
