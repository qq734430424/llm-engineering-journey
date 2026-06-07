"""
第3周 第2天：向量距离计算 + 语义检索
=====================================
从「手算两个向量的距离」→「给定查询，在文档库里搜 Top-K」

三个核心概念：
  1. 三种距离度量：余弦 / 欧氏 / 点积 —— 为什么语义检索用余弦
  2. 语义搜索 ≠ 关键词搜索 —— 措辞不同但意思相同也能命中
  3. Top-K 检索 —— 预编码文档 → 查询时只算一次 → 排序返回
"""

import math
import time
from typing import Callable

from sentence_transformers import SentenceTransformer

# ============================================================
# 1. 加载模型
# ============================================================
MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(f"✓ 模型已加载 ({MODEL.get_sentence_embedding_dimension()} 维)\n")


# ============================================================
# 2. 三种距离度量
# ============================================================
def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度 (-1~1)：方向越一致越接近 1。不受向量长度影响。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0


def euclidean(a: list[float], b: list[float]) -> float:
    """欧氏距离 (0~∞)：越接近 0 越相似。受向量长度影响。"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot_product(a: list[float], b: list[float]) -> float:
    """点积 (-∞~∞)：越大越相似。受向量长度和方向共同影响。"""
    return sum(x * y for x, y in zip(a, b))


# ============================================================
# 3. 语义检索引擎（极简版）
# ============================================================
class SemanticSearch:
    """
    极简语义检索引擎——去掉所有抽象，只保留核心逻辑。

    使用：
      ss = SemanticSearch(documents)
      results = ss.search("查询文本", top_k=3)
    """

    def __init__(self, documents: list[str]):
        self.documents = documents
        print(f"  编码 {len(documents)} 篇文档...", end=" ")
        t = time.time()
        self.vectors = MODEL.encode(documents, normalize_embeddings=True)
        print(f"完成 ({time.time() - t:.2f}s)")

    def search(
        self,
        query: str,
        top_k: int = 3,
        metric: Callable = cosine,
    ) -> list[tuple[float, str]]:
        """返回 top_k 条最相似的 (score, document)"""
        q_vec = MODEL.encode([query], normalize_embeddings=True)[0]

        scores = [
            (metric(q_vec, self.vectors[i]), self.documents[i])
            for i in range(len(self.documents))
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


# ============================================================
# 4. Demo
# ============================================================

# ---- 文档库（模拟一个企业内部知识库）----
docs = [
    "使用 Redis 缓存可以大幅降低数据库查询延迟",
    "Java 的 Spring Boot 框架适合构建微服务架构",
    "Kubernetes 是容器编排的事实标准",
    "Python 的 FastAPI 性能接近 Node.js，适合高并发场景",
    "数据库索引优化可以显著提升 SQL 查询速度",
    "Docker 容器化部署简化了环境配置流程",
    "前端 React 框架使用虚拟 DOM 提升渲染性能",
    "Elasticsearch 是一个分布式的全文搜索引擎",
    "微服务之间通过 gRPC 或消息队列进行通信",
    "使用 CI/CD 流水线可以自动化构建、测试和部署",
    "LangChain 是构建大模型应用的主流框架",
    "PostgreSQL 支持 JSON 查询和全文索引",
    "机器学习模型的部署通常使用模型服务化方案",
    "Nginx 反向代理可以负载均衡到多个后端服务",
    "Git 分支管理策略影响团队协作效率",
]

# 初始化检索引擎
print("=" * 65)
print("  语义检索引擎 Demo")
print("=" * 65)
ss = SemanticSearch(docs)

# ---- 测试 1：精确语义命中 vs 关键词匹配 ----
queries = [
    "怎么让数据库查询变快",
    "用什么技术部署微服务",
    "大模型应用开发框架",
    "前端页面渲染性能优化",
]

for q in queries:
    print(f"\n{'─' * 65}")
    print(f"  🔍 查询: \"{q}\"")
    print(f"{'─' * 65}")

    results = ss.search(q, top_k=3)
    for rank, (score, doc) in enumerate(results, 1):
        bar = "█" * int(score * 30)
        print(f"  #{rank}  [{score:.3f}]  {bar}")
        print(f"       {doc}")

# ---- 测试 2：三种距离度量对比 ----
print(f"\n{'=' * 65}")
print("  三种距离度量对比")
print(f"{'=' * 65}")
print(f"  查询: \"怎么提升系统性能\"")
print(f"  {'度量':10s} {'Top-1':40s} {'分数':>10s}")
print(f"  {'─' * 10} {'─' * 40} {'─' * 10}")

q = "怎么提升系统性能"
q_vec = MODEL.encode([q], normalize_embeddings=True)[0]

for name, fn in [("余弦", cosine), ("欧氏(反)", lambda a, b: -euclidean(a, b)), ("点积", dot_product)]:
    results = ss.search(q, top_k=1, metric=fn)
    score, doc = results[0]
    print(f"  {name:10s} {doc[:40]:40s} {score:>10.3f}")

print(f"\n  结论：三种度量给出的 Top-1 结果一致 → 用 cosine 就够了")
