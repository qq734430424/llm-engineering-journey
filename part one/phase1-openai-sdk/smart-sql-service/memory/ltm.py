"""长期记忆 - 带用户隔离的 Chroma 持久化"""

try:
    import chromadb
except ImportError:
    chromadb = None


class LongTermMemory:
    """用户级长期记忆管理

    功能：
    1. Schema 知识库（全局共享）
    2. 用户偏好记忆（按 user_id 隔离）
    3. 用户查询模式记忆（按 user_id 隔离）

    使用方式：
        ltm = LongTermMemory(chroma_path="./chroma_db")
        ltm.add_schema("users", "表 users: id, name, email...")
        schemas = ltm.query_schema("用户表")
        ltm.save_user_memory("user_A", "偏好简洁输出", mem_type="preference")
        prefs = ltm.query_user_memory("user_A", "偏好")
    """

    def __init__(self, chroma_path: str = "./chroma_db"):
        if chromadb is None:
            raise ImportError("请安装 chromadb: pip install chromadb")

        self.client = chromadb.PersistentClient(path=chroma_path)
        self.schema_col = self._get_or_create("global_schema")
        self.user_mem_col = self._get_or_create("user_memories")

    def _get_or_create(self, name: str):
        try:
            return self.client.get_collection(name)
        except ValueError:
            return self.client.create_collection(name)

    # ── Schema 检索（全局共享） ──

    def add_schema(self, table_name: str, schema_text: str):
        self.schema_col.add(
            documents=[schema_text],
            metadatas=[{"table": table_name}],
            ids=[f"schema_{table_name}_{id(table_name)}"],
        )

    def query_schema(self, query: str, top_k: int = 3) -> list[str]:
        results = self.schema_col.query(query_texts=[query], n_results=top_k)
        return results["documents"][0] if results.get("documents") else []

    # ── 用户级记忆（严格按 user_id 隔离） ──

    def save_user_memory(self, user_id: str, content: str,
                         mem_type: str = "query", metadata: dict = None):
        mem_id = f"{user_id}_{mem_type}_{id(content)}"
        meta = {"user_id": user_id, "type": mem_type}
        if metadata:
            meta.update(metadata)
        self.user_mem_col.add(
            documents=[content],
            metadatas=[meta],
            ids=[mem_id],
        )

    def query_user_memory(self, user_id: str, query: str,
                          mem_type: str = None, top_k: int = 3) -> list[str]:
        where = {"user_id": user_id}
        if mem_type:
            where["type"] = mem_type
        results = self.user_mem_col.query(
            query_texts=[query], n_results=top_k, where=where,
        )
        return results["documents"][0] if results.get("documents") else []

    def get_user_preferences(self, user_id: str) -> dict:
        prefs = self.query_user_memory(user_id, "偏好", mem_type="preference")
        parsed = {}
        for p in prefs:
            if "常用表" in p or "常用" in p:
                parsed["preferred_tables"] = p
            elif "格式" in p or "简洁" in p or "详细" in p:
                parsed["output_style"] = p
            else:
                parsed.setdefault("other", []).append(p)
        return parsed
