"""
第3周 第3天：Chroma 向量数据库入门
===================================
昨天你手动在内存里算距离 → 今天用 Chroma 存、搜、过滤。

Chroma 的三层抽象：
  Collection  ── 一张「表」，存同类型的向量
  Document    ── 原始文本
  Metadata    ── 每条文本的标签（来源、日期、分类），可以按元数据过滤

核心操作流程：
  1. 创建 Collection
  2. add(文档列表) → Chroma 自动编码 + 存储
  3. query(查询文本, top_n) → 返回最相似的文档
  4. 可选：按 metadata 过滤

与昨天的 SemanticSearch 对比：
  昨天：15 条文档全在内存，每次重启丢失
  Chroma：持久化到磁盘，支持数万条，带元数据过滤
"""

import time

import chromadb
from chromadb.utils import embedding_functions

# ============================================================
# 1. 初始化 Chroma
# ============================================================
print("=" * 65)
print("  Chroma 向量数据库 Demo")
print("=" * 65)

# Chroma 有两种模式：
#   PersistentClient   → 数据存磁盘，重启不丢
#   EphemeralClient    → 数据存内存，进程结束即丢（跟昨天一样）
client = chromadb.PersistentClient(path="./chroma_db")

# 指定 embedding 函数——Chroma 可以帮你自动编码
# 用跟昨天一样的 bge 模型
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5",
)

# ============================================================
# 2. 创建 Collection
# ============================================================
print("\n📦 创建 Collection: tech_knowledge ...")

# 如果已存在就删掉重建（Demo 用，生产环境不要）
try:
    client.delete_collection("tech_knowledge")
except Exception:
    pass

collection = client.create_collection(
    name="tech_knowledge",
    embedding_function=ef,
    metadata={"description": "技术知识库"},
)
print(f"   ✓ Collection 已创建 (ID: {collection.id})")


# ============================================================
# 3. 添加文档
# ============================================================
print("\n📝 添加文档 ...")
start = time.time()

documents = [
    "使用 Redis 缓存可以大幅降低数据库查询延迟，提升系统响应速度",
    "Java 的 Spring Boot 框架适合构建微服务架构，集成方便",
    "Kubernetes 是容器编排的事实标准，支持自动扩缩容",
    "Python 的 FastAPI 性能接近 Node.js，适合高并发 API 场景",
    "数据库索引优化可以显著提升 SQL 查询速度，B+树是最常用索引结构",
    "Docker 容器化部署简化了环境配置流程，实现一次构建到处运行",
    "React 前端框架使用虚拟 DOM 提升渲染性能，减少真实 DOM 操作",
    "Elasticsearch 是一个分布式的全文搜索引擎，基于 Lucene",
    "微服务之间通过 gRPC 或消息队列进行通信，实现解耦",
    "CI/CD 流水线可以自动化构建、测试和部署，提升交付效率",
    "LangChain 是构建大模型应用的主流框架，支持 RAG 和 Agent",
    "PostgreSQL 支持 JSON 查询和全文索引，适合混合查询场景",
    "Nginx 反向代理可以负载均衡到多个后端服务，提升系统可用性",
    "Git 分支管理策略影响团队协作效率，推荐 Git Flow 或 Trunk-Based",
]

# 每条文档配元数据——后面可以按元数据过滤
metadatas = [
    {"category": "性能", "level": "intermediate"},
    {"category": "架构", "level": "beginner"},
    {"category": "部署", "level": "advanced"},
    {"category": "性能", "level": "intermediate"},
    {"category": "数据库", "level": "advanced"},
    {"category": "部署", "level": "beginner"},
    {"category": "前端", "level": "intermediate"},
    {"category": "搜索", "level": "advanced"},
    {"category": "架构", "level": "advanced"},
    {"category": "DevOps", "level": "intermediate"},
    {"category": "AI", "level": "intermediate"},
    {"category": "数据库", "level": "advanced"},
    {"category": "部署", "level": "intermediate"},
    {"category": "DevOps", "level": "beginner"},
]

ids = [f"doc_{i}" for i in range(len(documents))]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)

elapsed = time.time() - start
print(f"   ✓ {len(documents)} 篇文档已入库 ({elapsed:.2f}s)")
print(f"   Collection 共 {collection.count()} 条记录")


# ============================================================
# 4. 语义查询
# ============================================================
def do_query(query: str, top_k: int = 3, where: dict = None):
    """语义查询 + 结果展示"""
    filter_desc = f" | 过滤: {where}" if where else ""
    print(f"\n{'─' * 65}")
    print(f"  🔍 \"{query}\"{filter_desc}")
    print(f"{'─' * 65}")

    kwargs = {"query_texts": [query], "n_results": top_k}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    for i, (doc_id, doc, meta, distance) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        # Chroma 返回的是距离（越小越近），不是相似度
        sim = 1 - distance  # 转成相似度方便看
        bar = "█" * int(sim * 30)
        print(f"  #{i+1}  [{sim:.3f}] {bar}")
        print(f"       {doc[:60]}")
        print(f"       📂 {meta['category']} · {meta['level']}")


# ---- 基础语义搜索 ----
do_query("怎么让数据库查询变快")
do_query("容器化部署用什么技术")
do_query("如何做前端性能优化")

# ---- 元数据过滤 ----
do_query("性能优化", where={"category": "性能"})      # 只看性能类
do_query("架构设计", where={"level": "advanced"})     # 只看高级内容

# ---- 混合过滤 ----
do_query("部署方案", where={
    "$and": [
        {"category": "部署"},
        {"level": {"$ne": "advanced"}},  # 排除高级内容
    ]
})

# ============================================================
# 5. 对比：关键词搜索 vs 语义搜索
# ============================================================
print(f"\n{'=' * 65}")
print("  关键词搜索 vs 语义搜索")
print(f"{'=' * 65}")
print(f"  查询: \"怎么让网站加载更快\"")
print(f"  文档库中没有任何一条包含「网站」或「加载」")

# 模拟关键词搜索——找不到
keyword_hit = [d for d in documents if "网站" in d or "加载" in d]
print(f"\n  关键词搜索: {len(keyword_hit)} 条结果")

# 语义搜索——能命中
results = collection.query(query_texts=["怎么让网站加载更快"], n_results=2)
print(f"  语义搜索:")
for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
    print(f"    #{i+1} [{1-distance:.3f}] {doc[:60]}")
