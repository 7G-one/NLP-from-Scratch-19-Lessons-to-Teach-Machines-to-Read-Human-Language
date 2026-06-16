import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第八章：语义相似度 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 实现同义词相似度查询
    2. 实现词向量余弦相似度
    3. 实现句子语义相似度计算

运行方式：
    python exercises.py
==============================================================================
"""

import math


# ==============================================================================
# 练习 1：实现同义词相似度查询
# ==============================================================================

# 简化的同义词词典（供练习使用）
EXERCISE_SYNONYM_DICT = {
    "开心": {"高兴", "快乐", "愉快"},
    "高兴": {"开心", "快乐", "愉快"},
    "快乐": {"开心", "高兴", "愉快"},
    "伤心": {"难过", "悲伤"},
    "难过": {"伤心", "悲伤"},
    "悲伤": {"伤心", "难过"},
    "电脑": {"计算机", "PC"},
    "计算机": {"电脑", "PC"},
    "猫": {"猫咪", "喵星人"},
    "狗": {"狗狗", "汪星人"},
}


def exercise_1_synonym_similarity(word1: str, word2: str) -> float:
    """
    练习 1：基于同义词典计算两个词的语义相似度

    ━━━━━━━ 生活类比 ━━━━━━━
    就像查字典：如果两个词在同一页（同一组同义词），它们就很相似。

    ━━━━━━━ 提示 ━━━━━━━
    1. 如果 word1 == word2，返回 1.0
    2. 如果 word1 在词典中，且 word2 是 word1 的同义词，返回 0.8
    3. 如果 word2 在词典中，且 word1 是 word2 的同义词，返回 0.8
    4. 如果两个词的同义词集合有交集，返回 0.5
    5. 否则返回 0.0

    参数：
        word1: 第一个词
        word2: 第二个词

    返回：
        语义相似度（0.0 到 1.0）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # if word1 == word2:
    #     return 1.0
    # if word1 in EXERCISE_SYNONYM_DICT and word2 in EXERCISE_SYNONYM_DICT[word1]:
    #     return 0.8
    # if word2 in EXERCISE_SYNONYM_DICT and word1 in EXERCISE_SYNONYM_DICT[word2]:
    #     return 0.8
    # syn1 = EXERCISE_SYNONYM_DICT.get(word1, set())
    # syn2 = EXERCISE_SYNONYM_DICT.get(word2, set())
    # if syn1 and syn2 and (syn1 & syn2):
    #     return 0.5
    # return 0.0
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：同义词相似度查询")
    print("=" * 60)

    test_cases = [
        ("开心", "高兴", 0.8),
        ("开心", "快乐", 0.8),
        ("开心", "伤心", 0.0),
        ("猫", "猫", 1.0),
        ("猫", "狗", 0.0),
    ]

    all_passed = True
    for w1, w2, expected in test_cases:
        result = exercise_1_synonym_similarity(w1, w2)
        if result is None:
            print("  [未完成] 请实现 exercise_1_synonym_similarity 函数")
            return False
        if abs(result - expected) < 0.01:
            print(f"  [正确] synonym_sim('{w1}', '{w2}') = {result:.2f}")
        else:
            print(f"  [错误] synonym_sim('{w1}', '{w2}'): 期望 {expected:.2f}, 实际 {result:.2f}")
            all_passed = False

    return all_passed


# ==============================================================================
# 练习 2：实现词向量余弦相似度
# ==============================================================================

# 简化的词向量（供练习使用）
EXERCISE_WORD_VECTORS = {
    "猫":   [0.9, 0.2, 0.0],
    "狗":   [0.9, 0.2, 0.1],
    "开心": [0.0, 0.9, 0.2],
    "伤心": [0.0, -0.9, 0.2],
    "电脑": [0.1, 0.0, 0.9],
}


def exercise_2_word_vector_similarity(word1: str, word2: str) -> float:
    """
    练习 2：基于词向量计算两个词的语义相似度

    ━━━━━━━ 生活类比 ━━━━━━━
    就像在地图上量两个地点的距离：
    - 猫和狗在地图上很近（都是宠物）
    - 猫和电脑在地图上很远（完全不同的东西）

    ━━━━━━━ 提示 ━━━━━━━
    1. 从 EXERCISE_WORD_VECTORS 中获取两个词的向量
    2. 如果某个词不在词典中，返回 0.0
    3. 计算点积：dot = sum(a * b for a, b in zip(vec1, vec2))
    4. 计算长度：norm = sqrt(sum(x^2))
    5. 返回 dot / (norm1 * norm2)

    参数：
        word1: 第一个词
        word2: 第二个词

    返回：
        语义相似度（-1 到 1 之间）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # vec1 = EXERCISE_WORD_VECTORS.get(word1)
    # vec2 = EXERCISE_WORD_VECTORS.get(word2)
    # if vec1 is None or vec2 is None:
    #     return 0.0
    # dot_product = sum(a * b for a, b in zip(vec1, vec2))
    # norm1 = math.sqrt(sum(a * a for a in vec1))
    # norm2 = math.sqrt(sum(b * b for b in vec2))
    # if norm1 == 0 or norm2 == 0:
    #     return 0.0
    # return dot_product / (norm1 * norm2)
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：词向量余弦相似度")
    print("=" * 60)

    test_cases = [
        ("猫", "狗", 0.99),      # 非常相似
        ("开心", "伤心", -0.85),  # 相反情感（负值）
        ("猫", "电脑", 0.15),    # 不太相似
    ]

    all_passed = True
    for w1, w2, expected in test_cases:
        result = exercise_2_word_vector_similarity(w1, w2)
        if result is None:
            print("  [未完成] 请实现 exercise_2_word_vector_similarity 函数")
            return False
        if abs(result - expected) < 0.1:
            print(f"  [正确] word_vec_sim('{w1}', '{w2}') = {result:.4f}")
        else:
            print(f"  [错误] word_vec_sim('{w1}', '{w2}'): 期望 {expected:.4f}, 实际 {result:.4f}")
            all_passed = False

    return all_passed


# ==============================================================================
# 练习 3：实现句子语义相似度
# ==============================================================================

def exercise_3_sentence_similarity(text1: str, text2: str) -> float:
    """
    练习 3：计算两个句子的语义相似度

    ━━━━━━━ 生活类比 ━━━━━━━
    比较两句话"说的是否是同一件事"：
    - "今天天气好" vs "今天气候不错" → 同一件事
    - "今天天气好" vs "苹果很好吃" → 不同的事

    ━━━━━━━ 提示 ━━━━━━━
    1. 将两个句子分别分词（可以用简单的 list(text)）
    2. 对于 text1 中的每个词，找 text2 中最相似的词
       - 先用同义词典（synonym_similarity）
       - 再用词向量（word_vector_similarity）
       - 取最大值
    3. 计算平均相似度
    4. 反向再算一次（text2 → text1）
    5. 返回两次的平均值

    参数：
        text1: 第一个句子
        text2: 第二个句子

    返回：
        句子语义相似度（0.0 到 1.0）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # # 简单分词：按字符
    # words1 = [c for c in text1 if c.strip() and c not in "，。！？ "]
    # words2 = [c for c in text2 if c.strip() and c not in "，。！？ "]
    # if not words1 or not words2:
    #     return 0.0
    #
    # def max_sim(word, word_list):
    #     best = 0.0
    #     for w in word_list:
    #         sim = exercise_1_synonym_similarity(word, w)
    #         if sim == 0:
    #             sim = max(0, exercise_2_word_vector_similarity(word, w))
    #         best = max(best, sim)
    #     return best
    #
    # # 正向
    # sim1 = sum(max_sim(w, words2) for w in words1) / len(words1)
    # # 反向
    # sim2 = sum(max_sim(w, words1) for w in words2) / len(words2)
    # return (sim1 + sim2) / 2
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：句子语义相似度")
    print("=" * 60)

    test_cases = [
        ("猫", "猫", 1.0, "完全相同"),
        ("猫", "狗", 0.5, "同类动物"),
        ("开心", "高兴", 0.5, "同义词"),
    ]

    all_passed = True
    for s1, s2, min_expected, desc in test_cases:
        result = exercise_3_sentence_similarity(s1, s2)
        if result is None:
            print("  [未完成] 请实现 exercise_3_sentence_similarity 函数")
            return False
        if result >= min_expected * 0.8:  # 允许 20% 的误差
            print(f"  [正确] '{s1}' vs '{s2}' ({desc}): {result:.4f}")
        else:
            print(f"  [错误] '{s1}' vs '{s2}' ({desc}): 期望 >= {min_expected:.2f}, 实际 {result:.4f}")
            all_passed = False

    return all_passed


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第八章 练习                    ║
    ║        语义相似度                                    ║
    ╚══════════════════════════════════════════════════════╝
    """)

    results = []
    results.append(("练习1: 同义词相似度", test_exercise_1()))
    results.append(("练习2: 词向量相似度", test_exercise_2()))
    results.append(("练习3: 句子语义相似度", test_exercise_3()))

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
        print("\n  所有练习完成！你已经掌握了语义相似度的核心技术。")
        print("  恭喜你完成了 G-one NLP 学院的全部课程！")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
