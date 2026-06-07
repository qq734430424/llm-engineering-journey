"""记忆中枢 - SQL Agent 的记忆入口，统合 STM + LTM"""
import json
import logging
import time

from memory.stm import ShortTermMemory
from memory.ltm import LongTermMemory

logger = logging.getLogger(__name__)


class MemoryHub:
    """记忆中枢

    SQL Agent 只需调用这一个类：
        hub = MemoryHub(db_url, chroma_path)
        hub.create_session(session_id, system_prompt)
        context = hub.get_context(session_id)    # → 全量历史
        prompt = hub.build_prompt(query)         # → 组合 Prompt
        hub.record(session_id, user_msg, assistant_msg)
    """

    def __init__(self, db_url: str = "sqlite:///sessions.db",
                 chroma_path: str = "./chroma_db"):
        self.stm = ShortTermMemory(db_url=db_url)
        try:
            self.ltm = LongTermMemory(chroma_path=chroma_path)
        except ImportError:
            logger.warning("ChromaDB 未安装，长期记忆功能不可用")
            self.ltm = None

    # ── 会话管理 ──

    def create_session(self, session_id: str, system_prompt: str):
        """创建新会话（幂等）"""
        self.stm.create(session_id, system_prompt)

    def get_context(self, session_id: str) -> list[dict]:
        """获取完整消息历史（可直接传给 LLM）"""
        return self.stm.get_context(session_id)

    def delete_session(self, session_id: str):
        """删除会话"""
        self.stm.delete(session_id)

    # ── 对话记录 ──

    def record(self, session_id: str,
               user_msg: dict, assistant_msg: dict,
               user_id: str = None):
        """记录一轮对话"""
        self.stm.append(session_id, user_msg, assistant_msg)

        # 尝试将用户查询写入长期记忆（异步友好）
        if self.ltm and user_id:
            try:
                query = user_msg.get("content", "")
                self.ltm.save_user_memory(
                    user_id, query, mem_type="query",
                    metadata={"session_id": session_id, "timestamp": time.time()},
                )
            except Exception as e:
                logger.debug(f"写入长期记忆失败: {e}")

    # ── Prompt 增强 ──

    def build_prompt(self, session_id: str, user_query: str,
                     user_id: str = None) -> str:
        """构建增强 Prompt（当前查询 + 历史上下文 + 长期记忆）"""
        parts = []

        # 1. 当前问题
        parts.append(f"## 当前问题\n{user_query}\n")

        # 2. 最近对话
        recent = self.stm.get_recent_context(session_id, rounds=3)
        if recent:
            lines = []
            for m in recent:
                role = "用户" if m["role"] == "user" else "助手"
                lines.append(f"{role}: {m['content']}")
            parts.append("## 最近对话\n" + "\n".join(lines) + "\n")

        # 3. 用户偏好（从长期记忆）
        if self.ltm and user_id:
            try:
                prefs = self.ltm.get_user_preferences(user_id)
                if prefs:
                    pref_lines = [f"{k}: {v}" for k, v in prefs.items()]
                    parts.append("## 用户偏好\n" + "\n".join(pref_lines) + "\n")
            except Exception:
                pass

        # 4. Schema 增强
        if self.ltm:
            try:
                schemas = self.ltm.query_schema(user_query, top_k=3)
                if schemas:
                    parts.append("## 相关表结构\n" + "\n".join(f"- {s}" for s in schemas))
            except Exception:
                pass

        return "\n\n".join(parts)
