"""
第3周 第4天：文档分块策略对比
================================
真实 RAG 场景：PDF 50 页 → 不能整本塞给 embedding → 必须切成小块。
切多大、怎么切，直接影响检索精度。

三种策略：
  固定大小   ── 每 N 个字符一刀，简单粗暴
  递归分割   ── 按段落/句子/字符优先级逐级切，保持语义完整
  语义分割   ── 模型自己判断"这里意思变了，该切了"（最智能，最慢）

Demo 做的事：
  同一段长文本 → 三种策略分别切 → 可视化对比每个 chunk 的语义完整性
"""

import re
from typing import Callable

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ============================================================
# 测试文本（模拟一篇技术博客）
# ============================================================
LONG_TEXT = """
# 微服务架构的演进与实践

## 1. 背景
随着业务规模的增长，传统的单体架构逐渐暴露出可维护性差、部署周期长、扩展困难等问题。
微服务架构通过将应用拆分为多个独立的服务，每个服务负责单一业务能力，从而解决了这些痛点。

## 2. 微服务核心组件

### 2.1 服务注册与发现
在微服务架构中，服务实例是动态变化的。Nacos和Consul是最常用的服务注册中心。
服务启动时向注册中心注册自己的地址，消费者通过服务名发现并调用，无需硬编码IP地址。

### 2.2 API 网关
API网关作为系统的统一入口，负责请求路由、限流、鉴权、日志收集等功能。
Spring Cloud Gateway 基于 WebFlux 构建，性能优于传统的 Zuul 网关。

### 2.3 配置中心
Nacos Config 和 Apollo 提供了集中化的配置管理能力，支持配置的实时推送和版本回滚。

## 3. 数据管理策略

### 3.1 数据库拆分
每个微服务拥有独立的数据库，避免跨服务直接访问数据库，通过API进行数据交互。
这带来了数据一致性的挑战，通常采用最终一致性和 Saga 模式来解决分布式事务问题。

### 3.2 缓存策略
Redis 作为分布式缓存，可以有效降低数据库压力。常见的缓存策略包括 Cache Aside、
Read/Write Through 和 Write Behind。缓存穿透、击穿、雪崩是必须处理的三大问题。

## 4. 通信方式

### 4.1 同步通信
RESTful API 和 gRPC 是最常用的同步通信方式。gRPC 基于 HTTP/2 和 Protobuf，
在性能上优于传统的 REST 调用，适合高频的内部服务通信。

### 4.2 异步通信
消息队列（RocketMQ、Kafka）用于服务间的异步解耦。典型场景包括订单创建后
通知库存系统扣减库存，以及用户注册后发送欢迎邮件等。

## 5. 部署与运维

### 5.1 容器化
Docker 容器化使得微服务的部署变得标准化。配合 Kubernetes 可以实现
自动扩缩容、滚动更新、服务自愈等高级特性。

### 5.2 可观测性
Prometheus 采集指标，Grafana 可视化展示，ELK 或 Loki 收集日志，
SkyWalking 做链路追踪——这是微服务可观测性的标准四件套。
"""

# ============================================================
# 策略1: 固定大小分块
# ============================================================
def fixed_size_split(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """固定大小 + 重叠窗口"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ============================================================
# 策略2: 递归分割（按优先级：段落 → 句子 → 字符）
# ============================================================
def recursive_split(text: str, chunk_size: int = 300) -> list[str]:
    """
    先按双换行（段落）切 → 如果段落太长，按句号切 → 还太长，按字符切。
    尽量保持语义边界不被打断。
    """
    paragraphs = text.split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 段落太长，按句子切
            sentences = re.split(r"(?<=[。！？])", para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += sent
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sent
            if current:
                chunks.append(current.strip())
    return chunks


# ============================================================
# 策略3: 模拟语义分割（用 embedding 检测语义边界）
# ============================================================
def semantic_split(text: str, model, threshold: float = 0.3) -> list[str]:
    """
    先按句子切，然后算相邻句子的语义相似度。
    相似度骤降的地方 → 话题变了 → 在这个位置切一刀。
    """
    sentences = re.split(r"(?<=[。！？\n])", text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if len(sentences) < 3:
        return ["\n".join(sentences)]

    vectors = model.encode(sentences, normalize_embeddings=True)

    # 算相邻句子相似度
    breakpoints = [0]
    for i in range(1, len(vectors)):
        sim = cosine_similarity([vectors[i-1]], [vectors[i]])[0][0]
        if sim < threshold:
            breakpoints.append(i)

    # 按断点合并
    chunks = []
    for start, end in zip(breakpoints, breakpoints[1:] + [len(sentences)]):
        chunks.append(" ".join(sentences[start:end]))
    return chunks


# ============================================================
# 对比展示
# ============================================================
print("=" * 70)
print("  文档分块策略对比")
print("=" * 70)
print(f"  原文长度: {len(LONG_TEXT)} 字符\n")

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

strategies = [
    ("固定大小 (200字符, 重叠50)", lambda: fixed_size_split(LONG_TEXT, 200, 50)),
    ("递归分割 (上限300字符)",     lambda: recursive_split(LONG_TEXT, 300)),
    ("语义分割 (相似度阈值0.3)",    lambda: semantic_split(LONG_TEXT, model, 0.3)),
]

for name, fn in strategies:
    chunks = fn()
    print(f"\n{'─' * 70}")
    print(f"  📦 {name} → {len(chunks)} 个块")
    print(f"{'─' * 70}")

    for i, chunk in enumerate(chunks):
        # 截断显示，标出片段开头
        preview = chunk[:100].replace("\n", " ")
        boundary = "│" if i < len(chunks) - 1 else "└"
        print(f"  {boundary} [{i:2d}] ({len(chunk)}字符) {preview}...")

# ============================================================
# 质量评估
# ============================================================
print(f"\n{'=' * 70}")
print("  策略优缺点总结")
print(f"{'=' * 70}")

comparison = """
┌───────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│           │ 固定大小             │ 递归分割             │ 语义分割             │
├───────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ 语义完整性 │ ❌ 可能在句子中间断  │ ✅ 保持段落/句子完整  │ ✅ 按话题边界切       │
│ 实现复杂度 │ ⭐ 最简单            │ ⭐⭐                 │ ⭐⭐⭐ 需要 embedding  │
│ 速度       │ 🚀 瞬时              │ 🚀 瞬时              │ 🐢 需要编码所有句子    │
│ 块大小均匀 │ ✅ 是                │ ⚠️ 不均匀             │ ❌ 差异很大           │
│ 适合场景   │ 快速原型             │ 通用文档             │ 技术文档/长报告       │
└───────────┴─────────────────────┴─────────────────────┴─────────────────────┘

推荐：
  入门 → 固定大小 300~500 + 重叠 50~100
  生产 → 递归分割（LangChain 的 RecursiveCharacterTextSplitter 就是这个）
  高精度长文档 → 递归 + 语义分割结合使用
"""
print(comparison)
