from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory


class StreamEvent(TypedDict):
    type: Literal["answer_delta", "retrieved_context"]
    text: str


def _extract_text(obj: Any) -> str:
    """从流式输出的各类对象中提取AI生成的文本内容。"""
    if obj is None:
        return ""
    # Prefer model-authored text only (skip tool/function/human messages).
    if isinstance(obj, BaseMessage):
        if isinstance(obj, AIMessage):
            content = getattr(obj, "content", None)
            if isinstance(content, str) and content.strip():
                return content
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if "messages" in obj:
            msgs = obj.get("messages")
            # Stream chunks often contain a list of messages, including ToolMessage.
            # Only surface AIMessage content to the CLI.
            if isinstance(msgs, (list, tuple)):
                for m in reversed(msgs):
                    s = _extract_text(m)
                    if s:
                        return s
                return ""
            return _extract_text(msgs)
        for k in ("output", "text", "final"):
            v = obj.get(k)
            s = _extract_text(v)
            if s:
                return s
        for v in obj.values():
            s = _extract_text(v)
            if s:
                return s
        return ""
    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            s = _extract_text(item)
            if s:
                return s
        return ""
    return ""


def _extract_tool_text(obj: Any) -> str:
    """从流式输出中提取工具消息文本（检索到的上下文）。"""
    if obj is None:
        return ""
    if isinstance(obj, BaseMessage):
        if isinstance(obj, ToolMessage):
            content = getattr(obj, "content", None)
            if isinstance(content, str) and content.strip():
                return content
        return ""
    if isinstance(obj, dict) and "messages" in obj:
        msgs = obj.get("messages")
        if isinstance(msgs, (list, tuple)):
            for m in reversed(msgs):
                s = _extract_tool_text(m)
                if s:
                    return s
    return ""


def stream_events(
    *,
    agent_with_memory,
    session_id: str,
    user_input: str,
) -> Iterator[StreamEvent]:
    """流式输出Agent的回答增量和检索到的上下文，分别以独立事件产出。"""
    last_answer = ""
    last_tool = ""
    for chunk in agent_with_memory.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"session_id": session_id}},
    ):
        tool_text = _extract_tool_text(chunk)
        if tool_text and tool_text != last_tool:
            yield {"type": "retrieved_context", "text": tool_text}
            last_tool = tool_text

        answer_text = _extract_text(chunk)
        if not answer_text:
            continue
        if answer_text != last_answer:
            yield {"type": "answer_delta", "text": answer_text[len(last_answer) :]}
            last_answer = answer_text


def make_redis_history_factory(*, url: str, key_prefix: str, ttl_seconds: int):
    """创建Redis聊天历史工厂函数，根据session_id获取对应的消息历史实例。"""

    def _factory(session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=session_id,
            url=url,
            key_prefix=key_prefix,
            ttl=ttl_seconds,
        )

    return _factory


def build_agent_with_memory(
    *,
    llm,
    tools: list,
    system_prompt: str,
    get_session_history: Callable[[str], Any],
):
    """创建带多轮对话历史记忆的LangChain Agent。"""
    agent = create_agent(llm, tools, system_prompt=system_prompt)
    return RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key="messages",
        history_messages_key="history",
    )


def stream_text(
    *,
    agent_with_memory,
    session_id: str,
    user_input: str,
) -> Iterable[str]:
    """流式输出Agent的文本回答增量（不含工具调用内容）。"""
    last = ""
    for chunk in agent_with_memory.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"session_id": session_id}},
    ):
        text = _extract_text(chunk)
        if not text:
            continue
        if text != last:
            yield text[len(last) :]
            last = text


def run_cli(
    *, agent_with_memory, session_id: str, clear_history_fn: Callable[[], None]
):
    """在命令行中运行RAG Agent交互循环，支持流式输出、清空历史和退出。"""
    try:
        from ollama import ResponseError as OllamaResponseError  # type: ignore
    except Exception:  # pragma: no cover
        OllamaResponseError = None  # type: ignore

    try:
        while True:
            user_input = input("\n你: ")
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() == "clear":
                clear_history_fn()
                print("历史已清空")
                continue
            if not user_input.strip():
                continue

            print("助手: ", end="", flush=True)
            try:
                printed_context = False
                for evt in stream_events(
                    agent_with_memory=agent_with_memory,
                    session_id=session_id,
                    user_input=user_input,
                ):
                    if evt["type"] == "retrieved_context":
                        if not printed_context:
                            print("\n\n【引用内容 / 检索到的上下文】\n", end="")
                            printed_context = True
                        print(evt["text"], end="\n\n", flush=True)
                        continue
                    print(evt["text"], end="", flush=True)
            finally:
                print()
    except Exception as e:
        if OllamaResponseError is not None and isinstance(e, OllamaResponseError):
            print(f"\n[Ollama错误] {e}")
            print(
                "提示：当前模型可能超过可用内存。请在 `config.py` 中改用更小/更高压缩的模型后重试。"
            )
            return
        print(f"\n[运行错误] {type(e).__name__}: {e}")
