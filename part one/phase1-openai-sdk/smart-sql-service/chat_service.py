"""
会话管理 + 多轮对话聊天服务
=============================
解决的问题：Postman 每次请求都是独立的，模型不记得上一轮说了什么。
用内存 session store 保存对话历史，前端只需传一个 session_id。
"""

import json
import logging
import time
import uuid
from typing import AsyncGenerator

from sqlalchemy import text

# 这些从 main.py 导入（循环引用的解决方式是等 main.py 启动后再注入）
logger = logging.getLogger(__name__)

# 由 main.py 注入
ENGINE = None
call_llm_with_retry = None
execute_sql = None
execute_write_sql = None
SYSTEM_PROMPT = None
EXECUTE_SQL_TOOL = None
EXECUTE_WRITE_SQL_TOOL = None

# ============================================================
# 会话存储（内存）
# ============================================================
# 生产环境换成 Redis，这里用 dict 足够
_sessions: dict[str, dict] = {}

SESSION_TTL = 3600  # 1 小时过期


def _clean_expired():
    """清理过期会话"""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s["updated_at"] > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
    if expired:
        logger.info(f"清理了 {len(expired)} 个过期会话")


def create_session() -> str:
    """创建新会话，返回 session_id"""
    _clean_expired()
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    return sid


def get_session(sid: str) -> dict | None:
    session = _sessions.get(sid)
    if session:
        session["updated_at"] = time.time()
    return session


def delete_session(sid: str):
    _sessions.pop(sid, None)


# ============================================================
# 聊天核心逻辑
# ============================================================
async def chat_stream(session_id: str, user_message: str) -> AsyncGenerator[str, None]:
    """
    多轮对话 + Function Calling + SSE 流式输出。

    流程：
    1. 拿到/创建会话 → 追加用户消息
    2. 调 LLM（带 tool 定义）
    3. 如果 LLM 调 tool → 执行 → 结果还给 LLM → 继续
    4. 最终回答流式推送给前端
    """
    session = get_session(session_id)
    if not session:
        session_id = create_session()
        session = get_session(session_id)

    messages = session["messages"]
    messages.append({"role": "user", "content": user_message})

    # ---- 第一轮：LLM 决定要不要调 tool ----
    yield _sse("status", "正在思考...")

    response = call_llm_with_retry(
        messages,
        tools=[EXECUTE_SQL_TOOL, EXECUTE_WRITE_SQL_TOOL],
        stream=False,
    )
    msg = response.choices[0].message

    # ---- 情况 A：模型直接回答（不需要调 tool）----
    if not msg.tool_calls:
        messages.append({"role": "assistant", "content": msg.content})
        session["updated_at"] = time.time()
        yield _sse("token", msg.content)
        yield _sse("done", "")
        return

    # ---- 情况 B：模型调了 tool ----
    tool_call = msg.tool_calls[0]
    func_name = tool_call.function.name
    sql = json.loads(tool_call.function.arguments)["sql"]

    # 推送给前端
    yield _sse("tool_call", json.dumps({"function": func_name, "sql": sql}, ensure_ascii=False))

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

    # 把 tool 调用和结果写回 message 历史
    messages.append({"role": "assistant", "tool_calls": [tool_call]})
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result,
    })

    yield _sse("status", "正在生成回答...")

    # ---- 第二轮：流式获取最终回答 ----
    stream = call_llm_with_retry(messages, stream=True)
    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_answer += delta.content
            yield _sse("token", delta.content)

    # 保存 assistant 的完整回答
    messages.append({"role": "assistant", "content": full_answer})
    session["updated_at"] = time.time()
    yield _sse("done", "")


def _sse(event_type: str, data: str) -> str:
    """构造 SSE 事件"""
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"
