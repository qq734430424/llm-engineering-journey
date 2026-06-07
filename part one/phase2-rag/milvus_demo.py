"""
第3周 扩展：Milvus 向量数据库实战
==================================
从 Chroma → Milvus，看看生产级向量数据库多了什么。

Chroma vs Milvus 核心差异：
  Chroma      → "pip install, 一句话 add，搞定。适合原型。"
  Milvus      → "先定义 Schema，指定索引类型，调参。适合生产。"

Milvus Lite   → 嵌入式模式，跟 Chroma 一样零配置，pip install 即用
Milvus Server → Docker 部署，支持分布式，百亿级向量

本 Demo 用 Milvus Lite，和你的 Chroma Demo 完全并行对比。
"""

import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pymilvus import (
    MilvusClient,
    DataType,
)

# ============================================================
# 1. 初始化
# ============================================================
print("=" * 65)
print("  Milvus 向量数据库 Demo (Lite 模式)")
print("=" * 65)

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
DIM = model.get_sentence_embedding_dimension()
print(f"  Embedding 维度: {DIM}")

# Milvus Lite —— 数据存本地文件 milvus_demo.db，跟 SQLite 一样
client = MilvusClient("milvus_demo.db")
print(f"  连接模式: 嵌入式 (Milvus Lite)\n")


# ============================================================
# 2. 定义 Schema —— Milvus 的最大特征
# ============================================================
print("📋 定义 Collection Schema ...")

# 每次运行用唯一名，避免 Windows 下 truncate/drop 的文件锁问题
COLLECTION_NAME = f"tech_knowledge_{datetime.now().strftime('%H%M%S')}"

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

# 主键
schema.add_field("id", DataType.INT64, is_primary=True)
# 原始文本
schema.add_field("text", DataType.VARCHAR, max_length=500)
# 元数据字段
schema.add_field("category", DataType.VARCHAR, max_length=50)
schema.add_field("level", DataType.VARCHAR, max_length=50)
# 向量字段
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)

# 创建 Collection + 指定索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 200},
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
)
print(f"   ✓ Collection 已创建: {COLLECTION_NAME} (HNSW, COSINE)\n")


# ============================================================
# 3. 插入数据 —— 和 Chroma 一样的 14 条文档
# ============================================================
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

categories = [
    "性能", "架构", "部署", "性能", "数据库",
    "部署", "前端", "搜索", "架构", "DevOps",
    "AI", "数据库", "部署", "DevOps",
]
levels = [
    "intermediate", "beginner", "advanced", "intermediate", "advanced",
    "beginner", "intermediate", "advanced", "advanced", "intermediate",
    "intermediate", "advanced", "intermediate", "beginner",
]

print("📝 生成向量 + 插入数据 ...")
start = time.time()

# 先生成 embedding
vectors = model.encode(documents, normalize_embeddings=True)

# 构造插入数据
data = []
for i, (doc, cat, lv, vec) in enumerate(zip(documents, categories, levels, vectors)):
    data.append({
        "id": i,
        "text": doc,
        "category": cat,
        "level": lv,
        "vector": vec.tolist(),
    })

client.insert(collection_name=COLLECTION_NAME, data=data)
print(f"   ✓ {len(data)} 条数据已入库 ({time.time() - start:.2f}s)")
print(f"   Collection 共 {client.get_collection_stats(COLLECTION_NAME)['row_count']} 条记录\n")


# ============================================================
# 4. 语义查询 —— 对比 Chroma 的 API
# ============================================================
def do_query(query: str, top_k: int = 3, filter_expr: str = None):
    """语义查询 + 条件过滤"""
    filter_desc = f" | 过滤: {filter_expr}" if filter_expr else ""
    print(f"{'─' * 65}")
    print(f"  🔍 \"{query}\"{filter_desc}")
    print(f"{'─' * 65}")

    # 编码查询向量
    q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

    # Chroma: collection.query(query_texts=[...])
    # Milvus: client.search(data=[...], anns_field="vector")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=top_k,
        filter=filter_expr,
        output_fields=["text", "category", "level"],
    )

    for i, hits in enumerate(results):
        for rank, hit in enumerate(hits):
            print(f" hit[distance] is : {hit['distance']}")
            sim = 1 - hit["distance"] if hit["distance"] <= 2 else hit["distance"]
            bar = "█" * int(sim * 30)
            entity = hit["entity"]
            print(f"  #{rank+1}  [{sim:.3f}] {bar}")
            print(f"       {entity['text'][:60]}")
            print(f"       📂 {entity['category']} · {entity['level']}")


# ---- 基础语义搜索 ----
do_query("怎么让数据库查询变快")
do_query("容器化部署用什么技术")

# ---- 元数据过滤 —— Milvus 用表达式字符串，不是 dict ----
# Chroma: where={"category": "性能"}
# Milvus: filter='category == "性能"'
do_query("性能优化", filter_expr='category == "性能"')
do_query("架构设计", filter_expr='level == "advanced"')
do_query("部署方案", filter_expr='category == "部署" and level != "advanced"')


# ============================================================
# 5. Chroma vs Milvus API 对照表
# ============================================================
print(f"\n{'=' * 65}")
print("  Chroma vs Milvus —— API 对照")
print(f"{'=' * 65}")

comparison = """
┌──────────────────┬─────────────────────────────────┬──────────────────────────────────┐
│ 操作             │ Chroma                          │ Milvus                           │
├──────────────────┼─────────────────────────────────┼──────────────────────────────────┤
│ 连接             │ chromadb.PersistentClient(path) │ MilvusClient("milvus.db")        │
│ 建 Collection    │ create_collection(name, ...)     │ 先定义 Schema → create_collection │
│ Schema           │ 无（自由字段）                    │ 严格（字段类型 + 索引）            │
│ 加数据           │ add(documents, metadatas, ids)  │ insert(data=[{id,text,vector}]) │
│ 自动编码         │ embedding_function=...           │ Schema 级 TEXTEMBEDDING Function  │
│ 语义查询         │ query(query_texts=[...])         │ search(data=[vector], anns_field)│
│ 元数据过滤       │ where={"category": "性能"}       │ filter='category == "性能"'      │
│ 索引类型         │ HNSW (固定)                      │ IVF_FLAT / HNSW / DiskANN 等 6种 │
│ 距离度量         │ L2 / COSINE / IP                │ L2 / COSINE / IP                 │
│ 分布式           │ ❌ 单机                          │ ✅ 原生分布式                      │
└──────────────────┴─────────────────────────────────┴──────────────────────────────────┘
"""
print(comparison)
