import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import numpy as np
except ImportError:
    print("本课需要 numpy 库，请运行：pip install numpy")
    print("安装后重新运行本文件即可。")
    exit(1)

"""
==============================================================================
第十二章：搜索引擎 — 完整演示
==============================================================================
G-one NLP 学院
日期：2026-05-16
==============================================================================
"""

from search_engine import (
    InvertedIndex, TFIDFSearchEngine, VectorSearchEngine,
    SimpleSearchEngine, explain_elasticsearch, preprocess
)


def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def lesson_inverted_index():
    """12.1 倒排索引原理与实现"""
    print_separator("12.1 倒排索引 — 搜索引擎的基石")

    print("""
    倒排索引就像图书馆的索引卡片系统：

    ┌──────────────────────────────────────────────┐
    │  正排索引（按文档找词）：                       │
    │    文档0 → ["机器", "学习", "人工智能"]         │
    │    文档1 → ["深度", "学习", "神经网络"]         │
    │                                              │
    │  倒排索引（按词找文档）：                       │
    │    "机器" → [文档0]                           │
    │    "学习" → [文档0, 文档1]                     │
    │    "深度" → [文档1]                           │
    │    "神经网络" → [文档1]                        │
    └──────────────────────────────────────────────┘
    """)

    # 实际演示
    index = InvertedIndex()

    docs = [
        "机器学习是人工智能的核心技术",
        "深度学习是机器学习的一个分支",
        "自然语言处理是人工智能的重要应用",
        "搜索引擎使用了倒排索引技术",
        "机器学习算法需要大量训练数据",
    ]

    print("  添加文档：")
    for i, doc in enumerate(docs):
        doc_id = index.add_document(doc)
        print(f"    文档{doc_id}: {doc}")

    # 显示索引
    index.display_index()

    # 搜索演示
    print("\n  搜索测试：")
    for query in ["机器学习", "人工智能", "搜索引擎", "神经网络"]:
        results = index.search(query)
        status = f"文档 {results}" if results else "无结果"
        print(f"    '{query}' → {status}")


def lesson_tfidf_ranking():
    """12.2 TF-IDF 搜索排序"""
    print_separator("12.2 TF-IDF 排序 — 让搜索结果更精准")

    print("""
    问题：搜索"机器学习"，返回了100篇文档，哪个最相关？

    解决方案：TF-IDF 评分

    ┌──────────────────────────────────────────────┐
    │  TF（词频）= 词在文档中出现的次数 / 文档总词数  │
    │    → 出现越多，越可能相关                      │
    │                                              │
    │  IDF（逆文档频率）= log(总文档数 / 包含该词的文档数) │
    │    → 越稀有的词，区分度越高                     │
    │                                              │
    │  TF-IDF = TF × IDF                           │
    │    → 综合考虑词频和区分度                      │
    └──────────────────────────────────────────────┘
    """)

    engine = TFIDFSearchEngine()

    docs = [
        "机器学习是人工智能的核心技术，机器学习需要大量数据",
        "深度学习是机器学习的一个重要分支",
        "自然语言处理是人工智能的重要应用领域",
        "搜索引擎使用倒排索引和TF-IDF技术",
        "机器学习算法包括决策树和支持向量机",
    ]

    print("  索引文档：")
    for doc in docs:
        engine.add_document(doc)

    query = "机器学习"
    results = engine.search(query)
    print(f"\n  查询: '{query}'")
    print(f"  排序结果：")
    for doc_id, score, text in results:
        print(f"    [{score:.4f}] 文档{doc_id}: {text}")


def lesson_vector_search():
    """12.3 向量空间模型搜索"""
    print_separator("12.3 向量空间模型 — 用数学衡量相似度")

    print("""
    核心思想：把文档变成向量，用余弦相似度比较

    生活类比：比较两个人的兴趣爱好
    ┌──────────────────────────────────────────────┐
    │  小明: [音乐=8, 电影=6, 运动=9, 游戏=2]       │
    │  小红: [音乐=7, 电影=8, 运动=3, 游戏=1]       │
    │  小刚: [音乐=1, 电影=2, 运动=8, 游戏=9]       │
    │                                              │
    │  小明和小红的兴趣更接近（余弦相似度更高）        │
    └──────────────────────────────────────────────┘
    """)

    engine = VectorSearchEngine()

    docs = [
        "机器学习算法需要大量训练数据来训练模型",
        "深度学习使用多层神经网络进行特征学习",
        "自然语言处理研究如何让计算机理解人类语言",
        "搜索引擎的核心技术是倒排索引和排序算法",
    ]

    for doc in docs:
        engine.add_document(doc)

    queries = ["深度学习神经网络", "搜索引擎排序", "机器学习算法"]
    for query in queries:
        results = engine.search(query, top_k=2)
        print(f"\n  查询: '{query}'")
        for doc_id, sim, text in results:
            print(f"    文档{doc_id} (相似度: {sim:.4f}): {text[:40]}...")


def lesson_elasticsearch():
    """12.4 Elasticsearch 概念入门"""
    print_separator("12.4 Elasticsearch — 工业级搜索引擎")

    print("""
    我们实现的是"迷你版"搜索引擎，工业界用的是 Elasticsearch：

    ┌──────────────────────────────────────────────┐
    │  我们的实现        │  Elasticsearch            │
    │  ──────────────────┼─────────────────────────│
    │  InvertedIndex     │  Shard（分片）            │
    │  add_document()    │  Index API               │
    │  search()          │  Search API              │
    │  TF-IDF            │  BM25（默认评分算法）      │
    │  preprocess()      │  Analyzer（分析器）        │
    └──────────────────────────────────────────────┘
    """)

    concepts = explain_elasticsearch()
    for name, explanation in concepts.items():
        print(f"\n  【{name}】")
        for line in explanation.strip().split("\n"):
            print(f"    {line.strip()}")


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          G-one NLP 学院 - 第十二章                        ║
    ║          搜索引擎                                        ║
    ╚══════════════════════════════════════════════════════════╝

    ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
    ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
    ███████╗█████╗  ███████║██████╔╝██║     ███████║
    ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
    ███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
    """)

    lesson_inverted_index()
    lesson_tfidf_ranking()
    lesson_vector_search()
    lesson_elasticsearch()

    print("\n" + "=" * 60)
    print("  第十二章 总结")
    print("=" * 60)
    print("""
    [OK] 倒排索引 — 词→文档 映射，搜索的基础
    [OK] TF-IDF 排序 — 用词频和逆文档频率计算相关度
    [OK] 向量空间模型 — 文档变向量，余弦相似度匹配
    [OK] Elasticsearch — 业界标准搜索引擎方案
    """)

    print("-" * 60)
    print("  下节预告：第十三章 — Word2Vec 词向量")
    print("-" * 60)
