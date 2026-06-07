"""短期记忆 - 三层存储：Redis 热窗口 → SQLite 持久化"""
import json
import time
from sqlalchemy import create_engine, text


class ShortTermMemory:
    """短期记忆管理器

    三层存储：
      L1: Redis 最近 N 轮对话（热窗口）
      L2: SQLite 完整对话历史（持久化）
      L3: 旧对话可压缩为摘要存向量库（另由 LTM 处理）

    使用方式：
        stm = ShortTermMemory(db_url="sqlite:///sessions.db")
        stm.append(session_id, user_msg, assistant_msg)
        history = stm.get_context(session_id)
    """

    MAX_HOT_ROUNDS = 5   # 保留最近 5 轮
    SESSION_TTL = 1800    # 30 分钟 Redis TTL

    def __init__(self, db_url: str = "sqlite:///sessions.db"):
        self.engine = create_engine(db_url)
        self._redis = None
        self._init_db()

    def _init_db(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS session_history (
                    session_id TEXT,
                    round_num INTEGER,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, round_num)
                )
            """))
            conn.commit()

    # ── 公开 API ──

    def get_context(self, session_id: str) -> list[dict]:
        """获取完整上下文"""
        return self._load_from_db(session_id)

    def get_recent_context(self, session_id: str, rounds: int = 3) -> list[dict]:
        """获取最近几轮（用于组合 Prompt）"""
        full = self._load_from_db(session_id)
        if not full:
            return []
        # 去掉第一条 system prompt
        user_assistant = [m for m in full if m["role"] in ("user", "assistant")]
        return user_assistant[-rounds * 2:]

    def append(self, session_id: str, user_msg: dict, assistant_msg: dict):
        """追加一轮对话"""
        self._save_to_db(session_id, user_msg, assistant_msg)

    def create(self, session_id: str, system_prompt: str):
        """创建新 session，写入 system prompt"""
        with self.engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM session_history WHERE session_id = :sid AND role = 'system'"),
                {"sid": session_id},
            ).fetchone()
            if exists:
                return  # 已存在
            conn.execute(
                text("INSERT INTO session_history (session_id, round_num, role, content) VALUES (:sid, 0, 'system', :c)"),
                {"sid": session_id, "c": system_prompt},
            )
            conn.commit()

    def delete(self, session_id: str):
        """删除 session"""
        with self.engine.connect() as conn:
            conn.execute(text("DELETE FROM session_history WHERE session_id = :sid"), {"sid": session_id})
            conn.commit()

    # ── 内部 ──

    def _load_from_db(self, session_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT role, content FROM session_history WHERE session_id = :sid ORDER BY round_num"),
                {"sid": session_id},
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def _save_to_db(self, session_id: str, user_msg: dict, assistant_msg: dict):
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT COALESCE(MAX(round_num), 0) FROM session_history WHERE session_id = :sid"),
                {"sid": session_id},
            )
            max_round = result.scalar()
            conn.execute(
                text(
                    "INSERT INTO session_history (session_id, round_num, role, content) "
                    "VALUES (:sid, :r1, :role1, :c1), (:sid, :r2, :role2, :c2)"
                ),
                {
                    "sid": session_id, "r1": max_round + 1,
                    "role1": user_msg["role"], "c1": user_msg["content"],
                    "r2": max_round + 2,
                    "role2": assistant_msg["role"], "c2": assistant_msg["content"],
                },
            )
            conn.commit()
