"""会话管理 + 多轮对话聊天服务
=============================
用 MemoryHub 替代原内存 dict，实现持久化短期/长期记忆。

升级点：
  - 对话历史存 SQLite（服务重启不丢）
  - 支持 user_id 用户级长期记忆（用户偏好 + 查询模式）
  - Schema 检索增强（Chroma 自动检索相关表结构）
  - 组合 Prompt（当前查询 + 历史上下文 + 用户偏好 + Schema）
"""

import json
import logging
import uuid
from typing import AsyncGenerator

from memory import MemoryHub

logger = logging.getLogger(__name__)

# 由 main.py 注入
ENGINE = None
call_llm_with_retry = None
execute_sql = None
execute_write_sql = None
SYSTEM_PROMPT = None
EXECUTE_SQL_TOOL = None
EXECUTE_WRITE_SQL_TOOL = None
DB_URL = "sqlite:///sessions.db"
CHROMA_PATH = "./chroma_db"

# ============================================================
# 持久化记忆中枢（替代原内存 _sessions dict）
# ============================================================
_hub = None


def _get_hub() -> MemoryHub:
    global _hub
    if _hub is None:
        _hub = MemoryHub(db_url=DB_URL, chroma_path=CHROMA_PATH)
    return _hub


def create_session() -> str:
    """创建新会话"""
    sid = uuid.uuid4().hex[:12]
    _get_hub().create_session(sid, SYSTEM_PROMPT)
    return sid


def get_session(sid: str) -> list[dict] | None:
    """获取会话历史"""
    ctx = _get_hub().get_context(sid)
    return ctx if ctx else None


def delete_session(sid: str):
    _get_hub().delete_session(sid)


# ============================================================
# 聊天核心逻辑
# ============================================================
async def chat_stream(session_id: str, user_message: str,
                      user_id: str = None) -> AsyncGenerator[str, None]:
    """
    多轮对话 + Function Calling + SSE 流式输出 + 记忆增强。

    与 v1 的区别：
      - 用 MemoryHub 替代内存 _sessions dict
      - 对话存 SQLite（持久化）
      - 支持 user_id 记录长期记忆
    """
    hub = _get_hub()

    # 1. 获取/创建会话
    history = hub.get_context(session_id)
    if not history:
        session_id = create_session()
        history = hub.get_context(session_id)

    # 2. 用户消息
    user_msg = {"role": "user", "content": user_message}
    history.append(user_msg)

    # 3. 组装 messages 传给 LLM（全量历史）
    messages = list(history)

    # ---- 第一轮：LLM 决定要不要调 tool ----
    yield _sse("status", "正在思考...")

    response = call_llm_with_retry(
        messages,
        tools=[EXECUTE_SQL_TOOL, EXECUTE_WRITE_SQL_TOOL],
        stream=False,
    )
    msg = response.choices[0].message

    # ---- 情况 A：直接回答（不调 tool）----
    if not msg.tool_calls:
        assistant_msg = {"role": "assistant", "content": msg.content}
        hub.record(session_id, user_msg, assistant_msg, user_id=user_id)

        words = msg.content
        for char in words:
            yield _sse("token", char)
        yield _sse("done", "")
        return

    # ---- 情况 B：调了 tool ----
    tool_call = msg.tool_calls[0]
    func_name = tool_call.function.name
    sql = json.loads(tool_call.function.arguments)["sql"]

    yield _sse("tool_call", json.dumps({"function": func_name, "sql": sql},
                                        ensure_ascii=False))

    # 执行 tool
    if func_name == "execute_sql":
        yield _sse("status", "正在执行查询...")
        rows = execute_sql(sql)
        tool_result = json.dumps(rows, ensure_ascii=False, default=str)
        yield _sse("query_result", tool_result)
    elif func_name == "execute_write_sql":
        yield _sse("status", "正在执行写操作...")
        try:
            result = execute_write_sql(sql, dry_run=False)
            tool_result = json.dumps(result, ensure_ascii=False)
            yield _sse("write_result", tool_result)
        except Exception as e:
            tool_result = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield _sse("error", tool_result)
    else:
        tool_result = '{"error": "未知的 tool"}'

    messages.append({"role": "assistant", "tool_calls": [tool_call]})
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result,
    })

    yield _sse("status", "正在生成回答...")

    # ---- 第二轮：流式最终回答 ----
    stream = call_llm_with_retry(messages, stream=True)
    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_answer += delta.content
            yield _sse("token", delta.content)

    assistant_msg = {"role": "assistant", "content": full_answer}
    hub.record(session_id, user_msg, assistant_msg, user_id=user_id)
    yield _sse("done", "")


def _sse(event_type: str, data: str) -> str:
    """构造 SSE 事件"""
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"
