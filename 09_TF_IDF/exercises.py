import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第九章：TF-IDF — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 实现 TF（词频）计算
    2. 实现 IDF（逆文档频率）计算
    3. 实现 TF-IDF 关键词提取

运行方式：
    python exercises.py
==============================================================================
"""

import math


# ==============================================================================
# 练习 1：实现 TF（词频）计算
# ==============================================================================

def exercise_1_compute_tf(term: str, document: list) -> float:
    """
    练习 1：计算词频（TF）

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你在数一篇作文中"爱"这个字出现了几次。
    如果作文有 100 个字，"爱" 出现了 10 次，
    那么 TF("爱") = 10 / 100 = 0.1

    ━━━━━━━ 提示 ━━━━━━━
    1. 统计 term 在 document 中出现的次数 count
    2. 获取 document 的总长度 total
    3. 返回 count / total
    4. 如果 document 为空，返回 0.0

    参数：
        term: 要查询的词
        document: 文档的分词结果列表

    返回：
        词频（0.0 到 1.0 之间）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # if not document:
    #     return 0.0
    # return document.count(term) / len(document)
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：TF（词频）计算")
    print("=" * 60)

    test_cases = [
        ("苹果", ["我", "喜欢", "吃", "苹果", "苹果"], 0.4),
        ("喜欢", ["我", "喜欢", "吃", "苹果", "苹果"], 0.2),
        ("香蕉", ["我", "喜欢", "吃", "苹果", "苹果"], 0.0),
        ("苹果", [], 0.0),
    ]

    all_passed = True
    for term, doc, expected in test_cases:
        result = exercise_1_compute_tf(term, doc)
        if result is None:
            print("  [未完成] 请实现 exercise_1_compute_tf 函数")
            return False
        if abs(result - expected) < 0.01:
            print(f"  [正确] TF('{term}', {doc}) = {result:.4f}")
        else:
            print(f"  [错误] TF('{term}', {doc}): 期望 {expected:.4f}, 实际 {result:.4f}")
            all_passed = False

    return all_passed


# ==============================================================================
# 练习 2：实现 IDF（逆文档频率）计算
# ==============================================================================

def exercise_2_compute_idf(term: str, documents: list) -> float:
    """
    练习 2：计算逆文档频率（IDF）

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你有 100 份报纸：
    - "股市" 只在 5 份报纸中出现 → IDF 高
    - "的" 在所有 100 份报纸中都出现 → IDF 低

    ━━━━━━━ 提示 ━━━━━━━
    1. 获取总文档数 N = len(documents)
    2. 统计包含 term 的文档数 df
       df = sum(1 for doc in documents if term in doc)
    3. 如果 df == 0，返回 0.0
    4. 返回 log(N / df)

    参数：
        term: 要查询的词
        documents: 文档集合（每个文档是分词结果列表）

    返回：
        IDF 值（非负数）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # n = len(documents)
    # df = sum(1 for doc in documents if term in doc)
    # if df == 0:
    #     return 0.0
    # return math.log(n / df)
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：IDF（逆文档频率）计算")
    print("=" * 60)

    corpus = [
        ["我", "喜欢", "苹果"],
        ["我", "喜欢", "运动"],
        ["今天", "天气", "好"],
        ["苹果", "手机", "好用"],
        ["我", "用", "苹果", "手机"],
    ]

    test_cases = [
        ("我", corpus, 0.5),      # 出现在 3/5 篇
        ("苹果", corpus, 0.2),    # 出现在 3/5 篇
        ("运动", corpus, 1.6),    # 只出现在 1/5 篇
        ("天气", corpus, 1.6),    # 只出现在 1/5 篇
    ]

    all_passed = True
    for term, docs, min_expected in test_cases:
        result = exercise_2_compute_idf(term, docs)
        if result is None:
            print("  [未完成] 请实现 exercise_2_compute_idf 函数")
            return False
        if result >= min_expected * 0.5:  # 宽松检查
            print(f"  [正确] IDF('{term}') = {result:.4f}")
        else:
            print(f"  [错误] IDF('{term}'): 期望 >= {min_expected:.4f}, 实际 {result:.4f}")
            all_passed = False

    return all_passed


# ==============================================================================
# 练习 3：实现 TF-IDF 关键词提取
# ==============================================================================

def exercise_3_extract_keywords(document: list, documents: list, top_k: int = 3) -> list:
    """
    练习 3：使用 TF-IDF 提取关键词

    ━━━━━━━ 生活类比 ━━━━━━━
    就像你在做读书笔记时，用荧光笔标出"重点词"。
    TF-IDF 就是自动帮你"画荧光笔"的工具。

    ━━━━━━━ 提示 ━━━━━━━
    1. 对 document 中的每个词，计算 TF：
       TF = count(word in document) / len(document)
    2. 对每个词计算 IDF：
       IDF = log(N / df)，其中 N = len(documents)
       df = sum(1 for doc in documents if word in doc)
    3. TF-IDF = TF * IDF
    4. 按 TF-IDF 降序排序
    5. 返回前 top_k 个 (词, TF-IDF值) 的列表

    参数：
        document: 目标文档的分词结果
        documents: 整个语料库（包含目标文档）
        top_k: 返回的关键词数量

    返回：
        关键词列表，每个元素是 (词, TF-IDF值)
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # n = len(documents)
    # tfidf_scores = {}
    # for word in set(document):
    #     tf = document.count(word) / len(document)
    #     df = sum(1 for doc in documents if word in doc)
    #     idf = math.log(n / df) if df > 0 else 0.0
    #     tfidf_scores[word] = tf * idf
    # sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    # return sorted_terms[:top_k]
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：TF-IDF 关键词提取")
    print("=" * 60)

    corpus = [
        ["自然", "语言", "处理", "是", "人工智能", "的", "重要", "方向"],
        ["机器", "学习", "是", "人工智能", "的", "核心", "技术"],
        ["深度", "学习", "在", "自然", "语言", "处理", "中", "应用", "广泛"],
    ]

    result = exercise_3_extract_keywords(corpus[0], corpus, top_k=3)

    if result is None:
        print("  [未完成] 请实现 exercise_3_extract_keywords 函数")
        return False

    print(f"\n  文档: {corpus[0]}")
    print(f"  Top-3 关键词:")
    for word, score in result:
        bar = "#" * int(score * 30)
        print(f"    {word:<8} {score:.4f}  {bar}")

    # 检查结果是否合理
    if len(result) == 3 and result[0][1] > 0:
        print("  [正确] 提取了 3 个关键词，且最高分大于 0")
        return True
    else:
        print("  [错误] 结果不正确")
        return False


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第九章 练习                    ║
    ║        TF-IDF                                       ║
    ╚══════════════════════════════════════════════════════╝
    """)

    results = []
    results.append(("练习1: TF 计算", test_exercise_1()))
    results.append(("练习2: IDF 计算", test_exercise_2()))
    results.append(("练习3: 关键词提取", test_exercise_3()))

    print("\n" + "=" * 60)
    print("  练习清单")
    print("=" * 60)
    for name, passed in results:
        status = "[完成]" if passed else "[未完成]"
        print(f"  {status} {name}")

    completed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  完成率: {completed}/{total}")

    if completed == total:
        print("\n  所有练习完成！你已经掌握了 TF-IDF 的核心技术。")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
