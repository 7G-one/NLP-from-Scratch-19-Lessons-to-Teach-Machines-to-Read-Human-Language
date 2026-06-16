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
第九章：TF-IDF — 完整演示
==============================================================================
G-one NLP 学院
日期：2026-05-16

运行方式：
    python main.py

前置知识：
    - 第二章：中文分词
    - 第七章：文本相似度

本章内容：
    1. TF（词频）概念与计算
    2. IDF（逆文档频率）概念与计算
    3. TF-IDF 综合应用
    4. sklearn TfidfVectorizer 实战
==============================================================================
"""

from tfidf import (
    compute_tf_ratio,
    compute_tf_log,
    compute_tf_augmented,
    compute_tf_document,
    compute_idf_raw,
    compute_idf_smooth,
    compute_idf_corpus,
    compute_tfidf,
    compute_tfidf_corpus,
    extract_keywords_tfidf,
    extract_keywords_all_documents,
    document_similarity_pair,
    simple_tokenize,
    print_tfidf_table,
    compare_tfidf_methods,
    sklearn_tfidf_demo,
    sklearn_tfidf_keywords,
)


def print_separator(title: str):
    """打印分隔线和标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def lesson_tf():
    """第一部分：TF（词频）"""

    print_separator("9.1 TF（词频，Term Frequency）")

    print("""
    TF 衡量一个词在一篇文档中出现的频率。

    核心思想：
      一个词在文档中出现的次数越多，它对这篇文档就越重要。

    生活类比：
      想象你在数一篇作文中每个字出现了几次。
      如果作文有 100 个字，"爱" 出现了 10 次，
      那么 TF("爱") = 10/100 = 0.1

    TF 的三种计算方法：
      1. 比例法：TF = 出现次数 / 总词数（最常用）
      2. 对数法：TF = log(1 + 出现次数)（压缩高频词的影响）
      3. 增强法：TF = 0.5 + 0.5 * 出现次数 / 最大词频（归一化）
    """)

    # 演示 1：三种 TF 计算方法
    print("-" * 40)
    print("示例 1：三种 TF 计算方法对比")

    document = ["我", "喜欢", "吃", "苹果", "苹果", "苹果", "很", "好吃", "苹果", "汁"]
    print(f"\n  文档: {document}")
    print(f"  文档长度: {len(document)}")

    terms = ["苹果", "喜欢", "我", "好吃"]
    print(f"\n  {'词':<6} {'比例法':>8} {'对数法':>8} {'增强法':>8}")
    print("  " + "-" * 35)
    for term in terms:
        ratio = compute_tf_ratio(term, document)
        log_tf = compute_tf_log(term, document)
        aug_tf = compute_tf_augmented(term, document)
        print(f"  {term:<6} {ratio:>8.4f} {log_tf:>8.4f} {aug_tf:>8.4f}")

    # 演示 2：TF 向量
    print("\n" + "-" * 40)
    print("示例 2：完整 TF 向量")

    tf_vector = compute_tf_document(document, method="ratio")
    print_tfidf_table(tf_vector, top_k=8)


def lesson_idf():
    """第二部分：IDF（逆文档频率）"""

    print_separator("9.2 IDF（逆文档频率，Inverse Document Frequency）")

    print("""
    IDF 衡量一个词在所有文档中的"稀有程度"。

    核心思想：
      如果一个词在很多文档中都出现，那它不太重要（比如"的"、"是"）。
      如果一个词只在少数文档中出现，那它很可能是关键词。

    生活类比：
      想象你有 100 份报纸：
      - "股市" 只在 5 份经济报纸中出现 → IDF 高 → 很有区分度
      - "的" 在所有 100 份报纸中都出现 → IDF 低 → 没有区分度

    IDF 的三种计算方法：
      1. 原始法：IDF = log(N / df)
      2. 平滑法：IDF = log((1+N) / (1+df)) + 1（最常用）
      3. 概率法：IDF = log((N-df) / df)
    """)

    # 模拟语料库
    corpus = [
        ["我", "喜欢", "吃", "苹果"],
        ["我", "喜欢", "运动"],
        ["今天", "天气", "很", "好"],
        ["苹果", "手机", "很", "好用"],
        ["我", "用", "苹果", "手机"],
    ]

    print("-" * 40)
    print("示例 1：IDF 计算")
    print(f"\n  语料库: {len(corpus)} 篇文档")
    for i, doc in enumerate(corpus):
        print(f"    文档 {i+1}: {doc}")

    # 计算各个词的 IDF
    terms = ["我", "苹果", "喜欢", "运动", "手机", "天气"]
    print(f"\n  {'词':<6} {'文档数(df)':>10} {'原始IDF':>10} {'平滑IDF':>10}")
    print("  " + "-" * 40)
    for term in terms:
        df = sum(1 for doc in corpus if term in doc)
        raw_idf = compute_idf_raw(term, corpus)
        smooth_idf = compute_idf_smooth(term, corpus)
        print(f"  {term:<6} {df:>10} {raw_idf:>10.4f} {smooth_idf:>10.4f}")

    print("""
    观察：
      - "我" 出现在 3 篇文档中 → IDF 较低
      - "运动" 只出现在 1 篇文档中 → IDF 较高
      - IDF 越高的词，区分度越强！
    """)

    # 演示 2：IDF 字典
    print("-" * 40)
    print("示例 2：完整 IDF 字典")

    idf_vector = compute_idf_corpus(corpus, method="smooth")
    sorted_idf = sorted(idf_vector.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  {'词':<8} {'IDF值':>8}  {'说明'}")
    print("  " + "-" * 40)
    for word, idf_val in sorted_idf[:10]:
        bar = "#" * int(idf_val * 5)
        note = "（稀有词，区分度高）" if idf_val > 2.0 else "（常见词，区分度低）"
        print(f"  {word:<8} {idf_val:>8.4f}  {bar} {note}")


def lesson_tfidf_comprehensive():
    """第三部分：TF-IDF 综合应用"""

    print_separator("9.3 TF-IDF 综合应用")

    print("""
    TF-IDF = TF * IDF

    生活类比：
      回到"一摞报纸"的例子：

      "股市" 在经济版面出现了 15 次：
        TF 高（出现多次） × IDF 高（只在少数报纸出现）→ TF-IDF 很高 → 关键词！

      "的" 在每份报纸中都出现了 200 次：
        TF 很高 × IDF 很低（每份报纸都有） → TF-IDF 接近 0 → 不是关键词！

    TF-IDF 的妙处：
      自动平衡了"出现频率"和"区分能力"两个因素。
    """)

    # 模拟语料库
    corpus = [
        ["自然", "语言", "处理", "是", "人工", "智能", "的", "重要", "方向"],
        ["机器", "学习", "是", "人工", "智能", "的", "核心", "技术"],
        ["深度", "学习", "在", "自然", "语言", "处理", "中", "应用", "广泛"],
        ["Python", "是", "最", "流行", "的", "编程", "语言"],
        ["数据", "挖掘", "和", "机器", "学习", "密切", "相关"],
    ]

    print("-" * 40)
    print("示例 1：每篇文档的关键词提取")

    all_keywords = extract_keywords_all_documents(corpus, top_k=3)

    doc_labels = [
        "自然语言处理是人工智能的重要方向",
        "机器学习是人工智能的核心技术",
        "深度学习在自然语言处理中应用广泛",
        "Python 是最流行的编程语言",
        "数据挖掘和机器学习密切相关",
    ]

    for i, (label, keywords) in enumerate(zip(doc_labels, all_keywords)):
        print(f"\n  文档 {i+1}: {label}")
        print(f"  关键词:")
        for word, score in keywords:
            bar = "#" * int(score * 30)
            print(f"    {word:<8} {score:.4f}  {bar}")

    # 演示 2：不同 TF-IDF 方法对比
    print("\n" + "-" * 40)
    print("示例 2：不同 TF-IDF 方法对比")
    compare_tfidf_methods(corpus[0], corpus)


def lesson_sklearn():
    """第四部分：sklearn TfidfVectorizer 实战"""

    print_separator("9.4 sklearn TfidfVectorizer 实战")

    print("""
    在实际项目中，我们通常使用 sklearn 的 TfidfVectorizer。

    它的优点：
      1. 一行代码完成 TF-IDF 计算
      2. 自动处理分词、停用词等
      3. 支持稀疏矩阵，节省内存
      4. 与其他 sklearn 模型无缝集成

    从零实现 vs sklearn：
      从零实现 → 理解原理
      sklearn   → 高效生产
    """)

    # 测试文档
    documents = [
        "自然语言处理是人工智能的重要方向",
        "机器学习是人工智能的核心技术",
        "深度学习在自然语言处理中应用广泛",
        "Python是最流行的编程语言",
        "数据挖掘和机器学习密切相关",
    ]

    print("-" * 40)
    print("示例 1：sklearn TF-IDF 矩阵")

    result = sklearn_tfidf_demo(documents)
    if result[0] is not None:
        tfidf_matrix, feature_names = result
        print(f"\n  文档数量: {len(documents)}")
        print(f"  特征数量: {len(feature_names)}")
        print(f"  矩阵形状: {tfidf_matrix.shape}")

        # 显示前几个特征
        print(f"\n  前 10 个特征: {list(feature_names[:10])}")

        # 显示第一篇文档的 TF-IDF 值
        print(f"\n  文档 1 的 TF-IDF 向量（非零值）:")
        doc1_tfidf = tfidf_matrix[0].toarray().flatten()
        for idx in doc1_tfidf.argsort()[-5:][::-1]:
            if doc1_tfidf[idx] > 0:
                print(f"    {feature_names[idx]}: {doc1_tfidf[idx]:.4f}")

    # 演示 2：关键词提取
    print("\n" + "-" * 40)
    print("示例 2：使用 sklearn 提取关键词")

    keywords_list = sklearn_tfidf_keywords(documents, top_k=3)
    if keywords_list:
        for i, (doc, keywords) in enumerate(zip(documents, keywords_list)):
            print(f"\n  文档: {doc}")
            for word, score in keywords:
                print(f"    {word}: {score:.4f}")


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第九章                        ║
    ║                                                      ║
    ║   ████████╗██████╗      ██╗██████╗ ███████╗          ║
    ║   ╚══██╔══╝██╔══██╗     ██║██╔══██╗██╔════╝          ║
    ║      ██║   ██████╔╝     ██║██║  ██║█████╗            ║
    ║      ██║   ██╔═══╝ ██   ██║██║  ██║██╔══╝            ║
    ║      ██║   ██║     ╚█████╔╝██████╔╝██║               ║
    ║      ╚═╝   ╚═╝      ╚════╝ ╚═════╝ ╚═╝               ║
    ║                                                      ║
    ║              词 频 - 逆 文 档 频 率                    ║
    ╚══════════════════════════════════════════════════════╝
    """)

    lesson_tf()
    lesson_idf()
    lesson_tfidf_comprehensive()
    lesson_sklearn()

    # 课程总结
    print("\n" + "=" * 60)
    print("  第九章 总结")
    print("=" * 60)
    print("""
    [OK] TF（词频，Term Frequency）
         衡量一个词在文档中出现的频率
         三种方法：比例法、对数法、增强法
         TF 高 → 在当前文档中重要

    [OK] IDF（逆文档频率，Inverse Document Frequency）
         衡量一个词在语料库中的"稀有程度"
         IDF 高 → 这个词在少数文档中出现 → 区分度强
         降低"停用词"的权重

    [OK] TF-IDF = TF * IDF
         综合"出现频率"和"区分能力"
         TF-IDF 高 → 这个词是当前文档的关键词
         最经典的文本特征提取方法之一

    [OK] sklearn TfidfVectorizer
         工业级实现，一行代码完成 TF-IDF 计算
         支持中文分词、停用词过滤等
         与其他 sklearn 模型无缝集成
    """)

    print("-" * 60)
    print("  下节预告：第十章 — 条件随机场（CRF）")
    print("-" * 60)
    print("""
    下一章我们将学习：
    - CRF 的基本概念和原理
    - 特征模板设计
    - CRF 在序列标注中的应用
    - sklearn-crfsuite 实战

    CRF 是序列标注的"终极武器"！
    """)
