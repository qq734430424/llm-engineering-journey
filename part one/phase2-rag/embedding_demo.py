"""
第3周 第1天：Embedding Demo（本地模型版）
===========================================
用 BAAI/bge-small-zh-v1.5 把文本转成向量，手算余弦相似度。

两组测试：
  组A ── 短技术术语（相关但不同义，预期 0.3~0.6）
  组B ── 同义句对（不同措辞说同一件事，预期 0.8+）
"""

import math
import time

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

print(f"📦 加载模型: {MODEL_NAME} ...")
start = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"   ✓ 加载完成 ({time.time() - start:.1f}s)")
print(f"   向量维度: {model.get_sentence_embedding_dimension()} 维\n")


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def show_similarity(name, texts):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    vectors = model.encode(texts, normalize_embeddings=True)

    # 矩阵
    print(f"\n{'':28s}", end="")
    for t in texts:
        print(f"{t[:10]:>10s}", end="")
    print()

    for i, t1 in enumerate(texts):
        print(f"{t1[:28]:28s}", end="")
        for j, t2 in enumerate(texts):
            sim = cosine_similarity(vectors[i], vectors[j])
            bar = "█" * int(sim * 40)
            print(f"  {sim*100:5.0f}", end="")
        print()

    # 第一项和其他的距离
    print(f"\n  与「{texts[0]}」的距离：")
    scores = [(i, cosine_similarity(vectors[0], vectors[i])) for i in range(1, len(texts))]
    scores.sort(key=lambda x: x[1], reverse=True)
    for idx, score in scores:
        if score > 0.75:
            label = "🟢 同义"
        elif score > 0.45:
            label = "🟡 相关"
        elif score > 0.25:
            label = "🟠 弱相关"
        else:
            label = "🔴 无关"
        print(f"    {score:.3f}  {label}  {texts[idx]}")


# ========== 组A：短术语（相关≠同义） ==========
show_similarity("组A：短技术术语 ——「相关」≠「同义」", [
    "Java后端开发工程师",
    "SpringBoot微服务架构",
    "今天天气真好，适合出去玩",
    "数据库索引优化与SQL调优",
    "Python机器学习与深度学习",
    "Redis缓存集群高可用方案",
])

# ========== 组B：同义句 ==========
show_similarity("组B：同义句对 —— 换个说法，模型认得出", [
    "今天天气真好，适合出去玩",
    "外面阳光明媚，出去逛逛吧",
    "数据库查询慢怎么办",
    "SQL 执行时间太长如何优化",
    "Java后端开发工程师",
    "Spring Boot 后端程序员",
])

print(f"\n{'='*70}")
print("  结论：")
print("  · 同义句 → 0.75+（模型认得出换个说法）")
print("  · 相关概念 → 0.40~0.60（同领域但不同义）")
print("  · 无关内容 → 0.25 以下（两种话题）")
print(f"{'='*70}")
