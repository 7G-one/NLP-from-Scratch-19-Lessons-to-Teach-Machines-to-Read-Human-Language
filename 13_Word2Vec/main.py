import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十三章：Word2Vec 词向量 — 完整演示
==============================================================================
G-one NLP 学院
日期：2026-05-16
==============================================================================
"""

try:
    import numpy as np
except ImportError:
    print("本课需要 numpy 库，请运行：pip install numpy")
    print("安装后重新运行本文件即可。")
    exit(1)

from word2vec import (
    explain_one_hot, build_vocabulary, generate_training_data,
    SimpleWord2Vec, SimpleSkipGram, word_analogy, demo_gensim_word2vec
)


def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def lesson_word_vector_concept():
    """13.1 词向量基础概念"""
    print_separator("13.1 什么是词向量")

    print("""
    问题：计算机怎么理解"国王"和"女王"意思接近？

    方案一：One-Hot 编码
    ┌──────────────────────────────────────────────┐
    │  "国王" → [1, 0, 0, 0, 0]                    │
    │  "女王" → [0, 1, 0, 0, 0]                    │
    │  相似度 = 0 （完全无法捕捉语义关系！）          │
    └──────────────────────────────────────────────┘

    方案二：Word2Vec 词向量
    ┌──────────────────────────────────────────────┐
    │  "国王" → [0.2, 0.8, 0.1, -0.3, 0.5]        │
    │  "女王" → [0.3, 0.7, 0.1, -0.2, 0.6]        │
    │  相似度 = 0.98 （语义相近！）                  │
    │  "苹果" → [-0.5, 0.1, 0.9, 0.4, -0.2]       │
    │  相似度 = 0.12 （语义不同）                    │
    └──────────────────────────────────────────────┘
    """)

    # 实际演示 One-Hot 的问题
    print("  One-Hot 编码的问题演示：")
    explain_one_hot()


def lesson_cbow_skipgram():
    """13.2 CBOW 和 Skip-gram 原理"""
    print_separator("13.2 CBOW 与 Skip-gram")

    print("""
    Word2Vec 有两种训练方式：

    ┌──────────────────────────────────────────────┐
    │  CBOW（完形填空）                              │
    │  输入: ["我", "喜欢", "苹果"]                  │
    │  预测: "吃"                                   │
    │  → 用上下文猜中间的词                          │
    ├──────────────────────────────────────────────┤
    │  Skip-gram（举一反三）                         │
    │  输入: "吃"                                   │
    │  预测: ["我", "喜欢", "苹果"]                  │
    │  → 看到一个词，猜它周围的词                    │
    └──────────────────────────────────────────────┘
    """)

    # 生成训练数据演示
    corpus = [
        ["我", "喜欢", "吃", "苹果"],
        ["他", "喜欢", "喝", "牛奶"],
    ]

    vocab, word_to_idx, _ = build_vocabulary(corpus)
    cbow_data, skipgram_data = generate_training_data(corpus, word_to_idx, window_size=2)

    print(f"  语料库: {corpus}")
    print(f"  词表: {vocab}")
    print(f"\n  CBOW 训练样本（共{len(cbow_data)}条）：")
    for ctx, target in cbow_data[:3]:
        ctx_words = [vocab[i] for i in ctx]
        print(f"    上下文 {ctx_words} → 目标 '{vocab[target]}'")

    print(f"\n  Skip-gram 训练样本（共{len(skipgram_data)}条）：")
    for center, ctx in skipgram_data[:3]:
        print(f"    中心词 '{vocab[center]}' → 上下文 '{vocab[ctx]}'")


def lesson_train_word2vec():
    """13.3 动手训练 Word2Vec"""
    print_separator("13.3 动手训练 Word2Vec")

    # 准备语料
    corpus = [
        ["我", "喜欢", "吃", "苹果"],
        ["我", "喜欢", "喝", "牛奶"],
        ["他", "喜欢", "吃", "香蕉"],
        ["她", "喜欢", "喝", "果汁"],
        ["我", "不喜欢", "吃", "辣椒"],
        ["他", "不喜欢", "喝", "咖啡"],
    ]

    vocab, word_to_idx, word_counts = build_vocabulary(corpus)
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    print(f"  词表大小: {len(vocab)}")
    print(f"  词表: {vocab}")

    # 训练 CBOW
    print("\n  --- 训练 CBOW 模型 ---")
    cbow_data, _ = generate_training_data(corpus, word_to_idx, window_size=2)
    cbow_model = SimpleWord2Vec(vocab_size=len(vocab), embedding_dim=50, learning_rate=0.01)
    cbow_model.train(cbow_data, epochs=100)

    # 展示结果
    print("\n  CBOW 训练结果：")
    print("  与 '喜欢' 最相似的词：")
    idx = word_to_idx["喜欢"]
    for word, sim in cbow_model.most_similar(idx, word_to_idx, idx_to_word, top_k=3):
        print(f"    {word}: {sim:.4f}")

    # 训练 Skip-gram
    print("\n  --- 训练 Skip-gram 模型 ---")
    _, skipgram_data = generate_training_data(corpus, word_to_idx, window_size=2)
    sg_model = SimpleSkipGram(vocab_size=len(vocab), embedding_dim=50, learning_rate=0.01)
    sg_model.train(skipgram_data, epochs=100)

    print("\n  Skip-gram 训练结果：")
    print("  词向量示例（前5维）：")
    for word in ["喜欢", "吃", "喝"]:
        if word in word_to_idx:
            vec = sg_model.get_word_vector(word_to_idx[word])
            print(f"    {word}: [{', '.join(f'{v:.3f}' for v in vec[:5])}...]")


def lesson_gensim_practice():
    """13.4 gensim Word2Vec 实战"""
    print_separator("13.4 gensim Word2Vec 实战")

    print("""
    在实际项目中，我们通常用 gensim 库来训练 Word2Vec：

    ━━━━━━━ 为什么用 gensim？ ━━━━━━━
    1. 高度优化，训练速度快
    2. 支持大语料库（内存映射）
    3. 内置模型保存/加载
    4. 支持增量训练
    5. API 简洁易用

    ━━━━━━━ 核心参数 ━━━━━━━
    vector_size: 词向量维度（通常 100-300）
    window: 上下文窗口大小（通常 5-10）
    min_count: 最小词频（过滤低频词）
    sg: 0=CBOW, 1=Skip-gram
    epochs: 训练轮数
    """)

    demo_gensim_word2vec()


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          G-one NLP 学院 - 第十三章                        ║
    ║          Word2Vec 词向量                                  ║
    ╚══════════════════════════════════════════════════════════╝

    ██╗    ██╗ ██████╗ ██████╗ ██████╗ ██████╗ ██╗   ██╗███████╗ ██████╗
    ██║    ██║██╔═══██╗██╔══██╗██╔══██╗╚════██╗██║   ██║██╔════╝██╔════╝
    ██║ █╗ ██║██║   ██║██████╔╝██████╔╝ █████╔╝██║   ██║█████╗  ██║
    ██║███╗██║██║   ██║██╔══██╗██╔══██╗ ╚═══██╗╚██╗ ██╔╝██╔══╝  ██║
    ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝ ╚████╔╝ ███████╗╚██████╗
     ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚══════╝ ╚═════╝
    """)

    lesson_word_vector_concept()
    lesson_cbow_skipgram()
    lesson_train_word2vec()
    lesson_gensim_practice()

    print("\n" + "=" * 60)
    print("  第十三章 总结")
    print("=" * 60)
    print("""
    [OK] 词向量 — 给每个词一个低维稠密向量（数字身份证）
    [OK] One-Hot 问题 — 高维稀疏、无法捕捉语义关系
    [OK] CBOW — 用上下文预测中心词（完形填空）
    [OK] Skip-gram — 用中心词预测上下文（举一反三）
    [OK] 词类比 — 国王 - 男人 + 女人 = 女王
    [OK] gensim — 工业级 Word2Vec 工具库
    """)

    print("-" * 60)
    print("  下节预告：第十四章 — 文本分类")
    print("-" * 60)
