"""
项目一 v2：智能 SQL 生成服务（生产级特性）
===========================================
v2 新增：
  1. 真正的 SSE 流式——LLM 吐一个字、前端看到一个字
  2. 分层错误处理——LLM超时 / SQL异常 / 参数校验，各走各的
  3. API Key 轮询——多 Key 轮转，单 Key 挂了自动切下一个

技术栈：FastAPI + OpenAI SDK + SQLAlchemy + SQLite
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import (
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

# ============================================================
# 配置
# ============================================================
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- 导入并注入聊天服务 ----
import chat_service  # noqa: E402

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="智能 SQL 生成服务 v2",
    description="自然语言查数据库——真正的流式输出 + 生产级错误处理",
    version="0.2.0",
)

# ============================================================
# 数据库
# ============================================================
ENGINE = create_engine("sqlite:///demo.db", echo=False)

# ============================================================
# 多供应商 API Key 轮询管理
# ============================================================
class KeyRotator:
    """
    多供应商 Key 轮询器。

    每次调用从 DeepSeek → MiniMax → Kimi 轮转。
    某家挂了自动跳过，5 分钟后恢复。

    .env 配置：
      DEEPSEEK_API_KEY=sk-xxx
      MINIMAX_API_KEY=sk-xxx
      KIMI_API_KEY=sk-xxx
    """

    PROVIDERS = [
        {
            "name": "DeepSeek",
            "env_key": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        },
        {
            "name": "MiniMax",
            "env_key": "MINIMAX_API_KEY",
            "base_url": "https://api.minimaxi.com/v1",
            "model": "minimax-text-01",
        },
        {
            "name": "Kimi",
            "env_key": "KIMI_API_KEY",
            "base_url": "https://api.moonshot.cn/v1",
            "model": "moonshot-v1-8k",
        },
    ]

    def __init__(self):
        self.providers: list[dict] = []
        for p in self.PROVIDERS:
            key = os.getenv(p["env_key"])
            if key:
                self.providers.append({**p, "api_key": key, "failed_at": None})

        self._index = 0

    @property
    def available_count(self) -> int:
        now = time.time()
        self._recover_stale(now)
        return sum(1 for p in self.providers if p["failed_at"] is None)

    @property
    def total_count(self) -> int:
        return len(self.providers)

    def _recover_stale(self, now: float):
        """5 分钟后自动恢复失败的 Key"""
        for p in self.providers:
            if p["failed_at"] and (now - p["failed_at"]) > 300:
                p["failed_at"] = None
                logger.info(f"{p['name']} Key 已自动恢复")

    def get_client(self) -> tuple[OpenAI, str]:
        """
        轮询获取下一个可用供应商的客户端 + 模型名。

        Returns:
            (OpenAI 客户端, 模型名)
        """
        if not self.providers:
            raise RuntimeError("未配置任何 API Key，请在 .env 中设置")

        now = time.time()
        self._recover_stale(now)

        attempts = 0
        while attempts < len(self.providers):
            p = self.providers[self._index % len(self.providers)]
            self._index += 1
            attempts += 1

            if p["failed_at"] is None:
                client = OpenAI(api_key=p["api_key"], base_url=p["base_url"])
                logger.debug(f"→ 使用 {p['name']} ({p['model']})")
                return client, p["model"]

        raise RuntimeError("所有供应商均已失败，请检查")

    def mark_failed(self, client: OpenAI):
        """标记当前供应商为失败"""
        key = client.api_key
        for p in self.providers:
            if p["api_key"] == key:
                p["failed_at"] = time.time()
                logger.warning(f"{p['name']} 已标记失败 (剩余可用: {self.available_count}/{self.total_count})")
                return

    def validate(self) -> list[str]:
        """启动时校验所有 Key"""
        results = []
        for p in self.providers:
            c = OpenAI(api_key=p["api_key"], base_url=p["base_url"])
            try:
                c.models.list()
                results.append(f"✓ {p['name']:8s} Key ...{p['api_key'][-6:]} 有效  ({p['base_url']})")
            except Exception as e:
                results.append(f"✗ {p['name']:8s} Key ...{p['api_key'][-6:]} 无效: {e}")
        return results


key_rotator = KeyRotator()


# ============================================================
# 自定义异常
# ============================================================
class LLMServiceError(Exception):
    """LLM API 错误基类"""

    pass


class SQLInjectionDetected(Exception):
    """检测到危险 SQL"""

    pass


# ============================================================
# LLM 调用封装（带重试）
# ============================================================
def call_llm_with_retry(messages, tools=None, stream=False, max_retries=3):
    """
    调用 LLM，自动处理：
    - Key 轮询
    - 超时重试
    - Rate Limit 退避
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            client, model = key_rotator.get_client()
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                stream=stream,
            )
        except AuthenticationError as e:
            key_rotator.mark_failed(client)
            last_error = e
            logger.error(f"认证失败 (attempt {attempt+1}/{max_retries})，切换 Key")
            continue
        except RateLimitError as e:
            last_error = e
            wait = (attempt + 1) * 2  # 2s, 4s, 6s
            logger.warning(f"限流 (attempt {attempt+1}/{max_retries})，{wait}s 后重试")
            time.sleep(wait)
            continue
        except APITimeoutError as e:
            last_error = e
            logger.warning(f"超时 (attempt {attempt+1}/{max_retries})，重试...")
            continue
        except APIError as e:
            last_error = e
            logger.error(f"API 错误 (status={e.status_code}): {e}")
            break  # 非重试类错误
        except Exception as e:
            last_error = e
            logger.error(f"未知错误: {e}")
            break

    raise LLMServiceError(f"LLM 调用失败（已重试 {max_retries} 次）: {last_error}")


# ============================================================
# 数据库 Schema
# ============================================================
DB_SCHEMA = """
-- 用户表
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    department VARCHAR(50),
    salary DECIMAL(10,2),
    hire_date DATE
);

-- 项目表
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    budget DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'planning',
    start_date DATE,
    end_date DATE
);

-- 用户-项目关联表
CREATE TABLE project_members (
    user_id INTEGER REFERENCES users(id),
    project_id INTEGER REFERENCES projects(id),
    role VARCHAR(30),
    PRIMARY KEY (user_id, project_id)
);
"""

# ---- 只读 Tool ----
EXECUTE_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "执行一条 SELECT 查询，返回数据。用于回答用户的数据查询问题。",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT 语句"}
            },
            "required": ["sql"],
        },
    },
}

# ---- 写操作 Tool ----
EXECUTE_WRITE_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_write_sql",
        "description": (
            "执行 INSERT/UPDATE/DELETE 写操作。"
            "注意：DELETE 和 UPDATE 必须包含 WHERE 条件。"
            "返回受影响的行数。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "INSERT/UPDATE/DELETE SQL 语句（DELETE/UPDATE 必须带 WHERE）"}
            },
            "required": ["sql"],
        },
    },
}

SYSTEM_PROMPT = f"""你是一个智能数据库助手，可以查询和修改数据。

数据库 Schema：
{DB_SCHEMA}

操作规则：
1. 查询数据 → 调用 execute_sql（只允许 SELECT）
2. 插入/更新/删除 → 调用 execute_write_sql
3. DELETE 和 UPDATE 必须带 WHERE 条件，禁止全表修改
4. 写操作前先向用户确认"即将执行: <SQL>，是否继续？"
5. 薪资/预算单位为万元
6. 回复简洁

禁止操作的表：无（所有表都可以操作）
禁止的关键字：DROP, ALTER, TRUNCATE"""


# ============================================================
# 数据模型
# ============================================================
class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=500, description="自然语言问题，如'开发部有哪些人？'"
    )


class QueryResponse(BaseModel):
    question: str
    sql: str | None = None
    data: list[dict] | None = None
    answer: str


class GenerateSQLRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class GenerateSQLResponse(BaseModel):
    question: str
    sql: str
    explanation: str


class WriteRequest(BaseModel):
    instruction: str = Field(
        ..., min_length=1, max_length=500,
        description="自然语言写操作，如'把张三的工资涨到3万'"
    )


class WriteResponse(BaseModel):
    instruction: str
    sql: str
    affected_rows: int
    message: str


class DryRunResponse(BaseModel):
    instruction: str
    sql: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


# ============================================================
# FastAPI 异常处理
# ============================================================
@app.exception_handler(LLMServiceError)
async def llm_error_handler(request: Request, exc: LLMServiceError):
    logger.error(f"LLM 服务错误: {exc}")
    return JSONResponse(
        status_code=502,
        content={"error": "LLM 服务不可用", "detail": str(exc)},
    )


@app.exception_handler(SQLInjectionDetected)
async def sql_injection_handler(request: Request, exc: SQLInjectionDetected):
    logger.warning(f"SQL 注入检测: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "不安全的 SQL 请求", "detail": str(exc)},
    )


@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "请求过于频繁，请稍后再试"},
    )


# ============================================================
# 安全校验（读写分离）
# ============================================================

# DDL 类关键字——永远禁止
FORBIDDEN_DDL = ["DROP", "ALTER", "TRUNCATE", "EXEC", "EXECUTE"]

# 允许的表（白名单）
ALLOWED_TABLES = ["users", "projects", "project_members"]

# DML 写操作关键字——允许但要加护栏
WRITE_OPERATIONS = ["INSERT", "UPDATE", "DELETE"]


def _extract_table(sql_upper: str) -> str | None:
    """从 SQL 中提取主表名"""
    import re
    # INSERT INTO table / UPDATE table / DELETE FROM table
    m = re.search(
        r"(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(\w+)",
        sql_upper, re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def _has_where(sql_upper: str) -> bool:
    """检查是否包含 WHERE 子句"""
    # 排除子查询中的 WHERE，只关心最外层
    # 简化规则：SQL 中有 WHERE 且在 INSERT 之后（UPDATE/DELETE 场景）
    return "WHERE" in sql_upper


def validate_write_sql(sql: str) -> str:
    """
    写操作安全校验（4 层护栏）。

    层1: 禁止 DDL
    层2: 表白名单
    层3: DELETE/UPDATE 必须有 WHERE
    层4: UPDATE 必须有 SET
    """
    upper = sql.strip().upper()
    sql_type = upper.split()[0] if upper.split() else ""

    # ---- 层1: 禁止 DDL ----
    for kw in FORBIDDEN_DDL:
        if kw in upper:
            raise SQLInjectionDetected(f"禁止 DDL 操作: {kw}")

    # 只允许 INSERT / UPDATE / DELETE
    if sql_type not in WRITE_OPERATIONS:
        raise SQLInjectionDetected(
            f"写操作仅支持 INSERT/UPDATE/DELETE，收到: {sql_type}"
        )

    # ---- 层2: 表白名单 ----
    table = _extract_table(upper)
    if not table:
        raise SQLInjectionDetected("无法识别操作的目标表")
    if table not in ALLOWED_TABLES:
        raise SQLInjectionDetected(
            f"表 '{table}' 不在白名单中。允许的表: {ALLOWED_TABLES}"
        )

    # ---- 层3: DELETE/UPDATE 必须有 WHERE ----
    if sql_type in ("DELETE", "UPDATE"):
        if not _has_where(upper):
            raise SQLInjectionDetected(
                f"{sql_type} 必须包含 WHERE 子句，禁止全表操作"
            )

    # ---- 层4: UPDATE 必须有 SET ----
    if sql_type == "UPDATE" and "SET" not in upper:
        raise SQLInjectionDetected("UPDATE 必须包含 SET 子句")

    return sql.strip()


def validate_read_sql(sql: str) -> str:
    """校验只读 SQL"""
    upper = sql.strip().upper()
    if not upper.startswith("SELECT"):
        raise SQLInjectionDetected(f"只允许 SELECT 语句，收到: {sql[:50]}")
    for kw in FORBIDDEN_DDL:
        if kw in upper:
            raise SQLInjectionDetected(f"SQL 包含禁止关键字: {kw}")
    return sql.strip()


def execute_sql(sql: str) -> list[dict]:
    """安全执行只读 SQL"""
    safe_sql = validate_read_sql(sql)
    try:
        with ENGINE.connect() as conn:
            result = conn.execute(text(safe_sql))
            return [dict(row._mapping) for row in result]
    except Exception as e:
        logger.error(f"SQL 执行失败: {sql[:100]} → {e}")
        raise HTTPException(status_code=400, detail=f"SQL 执行失败: {e}")


def execute_write_sql(sql: str, dry_run: bool = False) -> dict:
    """
    安全执行写操作。

    事务流程: BEGIN → 执行 → 检查 affected rows → 决定 COMMIT/ROLLBACK
    dry_run=True 时不提交，只返回 SQL 和预计影响。
    """
    safe_sql = validate_write_sql(sql)

    if dry_run:
        # 干跑：只校验不执行
        return {
            "sql": safe_sql,
            "dry_run": True,
            "message": "SQL 校验通过（未执行）",
            "affected_rows": None,
        }

    try:
        with ENGINE.connect() as conn:
            # SQLite 默认 autocommit，需要手动开启事务
            trans = conn.begin()
            try:
                result = conn.execute(text(safe_sql))
                affected = result.rowcount
                # 更新操作可能 rowcount 为 -1（SQLite 特性），用 changes() 获取
                if affected < 0:
                    affected = conn.execute(text("SELECT changes()")).scalar()

                trans.commit()
                logger.info(f"写操作成功: {affected} 行受影响 → {safe_sql[:80]}")
                return {
                    "sql": safe_sql,
                    "dry_run": False,
                    "message": f"执行成功，{affected} 行受影响",
                    "affected_rows": affected,
                }
            except Exception:
                trans.rollback()
                raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"写操作失败: {safe_sql[:80]} → {e}")
        raise HTTPException(status_code=400, detail=f"写操作失败: {e}")


# ============================================================
# 初始化数据库
# ============================================================
def init_database():
    with ENGINE.connect() as conn:
        result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        )
        if result.fetchone():
            logger.info("→ 表已存在，跳过数据库初始化")
            return

        for stmt in DB_SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt and "CREATE TABLE" in stmt:
                conn.execute(text(stmt))

        # 插入测试数据
        conn.execute(text("""
            INSERT INTO users VALUES
            (1, '张三', 'zhangsan@example.com', '开发部', 2.5, '2020-03-15'),
            (2, '李四', 'lisi@example.com', '开发部', 3.0, '2019-07-01'),
            (3, '王五', 'wangwu@example.com', '产品部', 2.8, '2021-01-10'),
            (4, '赵六', 'zhaoliu@example.com', '产品部', 2.2, '2022-06-20'),
            (5, '孙七', 'sunqi@example.com', '测试部', 1.8, '2023-02-28'),
            (6, '周八', 'zhouba@example.com', '开发部', 4.0, '2018-05-12')
        """))
        conn.execute(text("""
            INSERT INTO projects VALUES
            (1, '智能客服系统', 200.00, 'active', '2025-01-01', '2025-12-31'),
            (2, '数据中台', 500.00, 'active', '2025-03-01', '2026-06-30'),
            (3, '移动端App', 150.00, 'planning', '2026-01-01', '2026-09-30'),
            (4, '内部OA系统', 80.00, 'completed', '2024-06-01', '2025-03-31')
        """))
        conn.execute(text("""
            INSERT INTO project_members VALUES
            (1, 1, 'developer'), (1, 2, 'tech_lead'),
            (2, 1, 'developer'), (2, 2, 'developer'),
            (3, 2, 'product_manager'), (4, 3, 'product_manager'),
            (6, 1, 'architect'), (6, 2, 'architect')
        """))
        conn.commit()
    logger.info("✓ 数据库初始化完成")


# ============================================================
# 核心：非流式 NL → SQL → 回答
# ============================================================
def run_nl_to_sql(question: str) -> tuple[str | None, list[dict] | None, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # 第一轮：模型决定是否调用 execute_sql
    response = call_llm_with_retry(messages, tools=[EXECUTE_SQL_TOOL])
    msg = response.choices[0].message

    # 不需要查库
    if not msg.tool_calls:
        return None, None, msg.content

    # 需要查库
    tool_call = msg.tool_calls[0]
    sql = json.loads(tool_call.function.arguments)["sql"]

    rows = execute_sql(sql)

    # 第二轮：用结果让模型解释
    messages.append({"role": "assistant", "tool_calls": [tool_call]})
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(rows, ensure_ascii=False, default=str),
    })

    final = call_llm_with_retry(messages)
    return sql, rows, final.choices[0].message.content


# ============================================================
# 核心：非流式 NL → 写操作
# ============================================================
def run_nl_to_write(instruction: str, dry_run: bool = False) -> dict:
    """
    NL → LLM 生成 INSERT/UPDATE/DELETE → 四层安全校验 → 事务执行。

    模型拿到两个 tool: execute_sql（查数据）和 execute_write_sql（写数据）。
    流程：LLM 自主决定先查后改，还是直接改。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    # 给模型两个 tool：它可以先查再改
    response = call_llm_with_retry(
        messages, tools=[EXECUTE_SQL_TOOL, EXECUTE_WRITE_SQL_TOOL]
    )
    msg = response.choices[0].message

    if not msg.tool_calls:
        return {"sql": "", "affected_rows": 0, "message": msg.content, "dry_run": dry_run}

    tool_call = msg.tool_calls[0]
    func_name = tool_call.function.name
    sql = json.loads(tool_call.function.arguments)["sql"]

    # 如果是只读操作（模型选择先查询确认）
    if func_name == "execute_sql":
        rows = execute_sql(sql)
        messages.append({"role": "assistant", "tool_calls": [tool_call]})
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(rows, ensure_ascii=False, default=str),
        })
        # 继续让模型判断，这次它应该选 execute_write_sql
        response2 = call_llm_with_retry(
            messages, tools=[EXECUTE_WRITE_SQL_TOOL]
        )
        msg2 = response2.choices[0].message
        if not msg2.tool_calls:
            return {"sql": sql, "affected_rows": 0, "message": msg2.content, "dry_run": dry_run}
        tool_call = msg2.tool_calls[0]
        sql = json.loads(tool_call.function.arguments)["sql"]

    # 执行写操作
    result = execute_write_sql(sql, dry_run=dry_run)
    return {**result, "instruction": instruction}


# ============================================================
# 核心：真正的 SSE 流式 NL → SQL → 回答
# ============================================================
async def run_nl_to_sql_stream(question: str) -> AsyncGenerator[str, None]:
    """
    真正的流式：三步走，每一步都实时推送给前端。

    阶段1: "status" - "正在分析问题..."
    阶段2: "sql"    - 生成的 SQL
    阶段3: "token"  - 自然语言回答，逐 token 推送
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # ---- 阶段 1：模型决定是否调 tool ----
    yield _sse_event("status", "正在分析问题...")

    response = call_llm_with_retry(messages, tools=[EXECUTE_SQL_TOOL])
    msg = response.choices[0].message

    # 不需要查库 → 直接返回文本
    if not msg.tool_calls:
        yield _sse_event("answer", msg.content)
        yield _sse_event("done", "")
        return

    # ---- 阶段 2：执行 SQL ----
    tool_call = msg.tool_calls[0]
    sql = json.loads(tool_call.function.arguments)["sql"]

    yield _sse_event("sql", sql)
    yield _sse_event("status", "正在执行查询...")

    rows = execute_sql(sql)
    yield _sse_event("data", json.dumps(rows, ensure_ascii=False, default=str))
    yield _sse_event("status", f"查到 {len(rows)} 条记录，正在生成回答...")

    # ---- 阶段 3：流式获取最终回答 ----
    messages.append({"role": "assistant", "tool_calls": [tool_call]})
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(rows, ensure_ascii=False, default=str),
    })

    # 真正的流式调用！
    stream = call_llm_with_retry(messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield _sse_event("token", delta.content)

    yield _sse_event("done", "")


def _sse_event(event_type: str, data: str) -> str:
    """构造 SSE 事件"""
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


# ============================================================
# API 路由
# ============================================================

@app.post("/api/query", response_model=QueryResponse)
def api_query(req: QueryRequest):
    """普通查询——等模型完全回答后一次性返回"""
    try:
        sql, data, answer = run_nl_to_sql(req.question)
    except LLMServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return QueryResponse(question=req.question, sql=sql, data=data, answer=answer)


@app.post("/api/query/stream")
async def api_query_stream(req: QueryRequest):
    """
    流式查询——真正的 SSE 推送。
    前端只需监听 EventSource，收到 'token' 事件就追加文本。
    """
    return StreamingResponse(
        run_nl_to_sql_stream(req.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 告诉 Nginx 不要缓冲
        },
    )


@app.post("/api/generate-sql", response_model=GenerateSQLResponse)
def api_generate_sql(req: GenerateSQLRequest):
    """只生成 SQL，不执行"""
    try:
        response = call_llm_with_retry([
            {
                "role": "system",
                "content": f"你是一个 SQL 专家。根据需求生成 SQL SELECT 语句。\n{DB_SCHEMA}\n只输出 SQL 和简短解释。",
            },
            {"role": "user", "content": req.question},
        ])
    except LLMServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    text = response.choices[0].message.content
    parts = text.split("\n", 1)
    sql = parts[0].strip().strip("`").lstrip("sql")
    explanation = parts[1].strip() if len(parts) > 1 else ""
    return GenerateSQLResponse(question=req.question, sql=sql, explanation=explanation)


@app.post("/api/write", response_model=WriteResponse)
def api_write(req: WriteRequest):
    """自然语言写操作——INSERT / UPDATE / DELETE"""
    try:
        result = run_nl_to_write(req.instruction, dry_run=False)
    except LLMServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except SQLInjectionDetected as e:
        raise HTTPException(status_code=400, detail=str(e))

    return WriteResponse(
        instruction=req.instruction,
        sql=result["sql"],
        affected_rows=result["affected_rows"],
        message=result["message"],
    )


@app.post("/api/write/dry-run", response_model=DryRunResponse)
def api_write_dry_run(req: WriteRequest):
    """写操作干跑——只校验 SQL 安全性，不实际执行"""
    try:
        result = run_nl_to_write(req.instruction, dry_run=True)
    except LLMServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except SQLInjectionDetected as e:
        raise HTTPException(status_code=400, detail=str(e))

    return DryRunResponse(
        instruction=req.instruction,
        sql=result["sql"],
        message=result["message"],
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_keys": key_rotator.available_count,
        "total_providers": key_rotator.total_count,
    }


# ============================================================
# 聊天服务（多轮对话 + 会话管理）
# ============================================================

# 依赖注入
chat_service.ENGINE = ENGINE
chat_service.call_llm_with_retry = call_llm_with_retry
chat_service.execute_sql = execute_sql
chat_service.execute_write_sql = execute_write_sql
chat_service.SYSTEM_PROMPT = SYSTEM_PROMPT
chat_service.EXECUTE_SQL_TOOL = EXECUTE_SQL_TOOL
chat_service.EXECUTE_WRITE_SQL_TOOL = EXECUTE_WRITE_SQL_TOOL
chat_service.DB_URL = f"sqlite:///{Path(__file__).parent / 'sessions.db'}"
chat_service.CHROMA_PATH = str(Path(__file__).parent / "chroma_db")


@app.post("/api/chat/session")
def create_chat_session():
    """创建新的聊天会话"""
    sid = chat_service.create_session()
    return {"session_id": sid}


@app.post("/api/chat/stream")
async def api_chat_stream(req: Request):
    """
    多轮对话流式端点。

    Body: {"session_id": "...", "message": "..."}
    返回 SSE 流: status / tool_call / query_result / write_result / token / done
    """
    body = await req.json()
    session_id = body.get("session_id", "")
    message = body.get("message", "")

    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    return StreamingResponse(
        chat_service.chat_stream(session_id, message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# 挂载静态文件 + 首页
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn

    # 启动前校验 Key
    print("\n🔑 API Key 校验结果:")
    for result in key_rotator.validate():
        print(f"  {result}")
    if key_rotator.available_count == 0:
        print("\n❌ 没有可用的 API Key，请在 .env 中配置 DEEPSEEK_API_KEY")
        exit(1)

    init_database()
    print(f"\n{'='*50}")
    print("  智能 SQL 生成服务 v2 已启动")
    print("  API 文档: http://localhost:8000/docs")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
