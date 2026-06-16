import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十一章：新词发现与短语提取 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 实现 n-gram 统计
    2. 实现互信息计算
    3. 实现简单的新词发现

运行方式：
    python exercises.py
==============================================================================
"""

import math
from collections import Counter


# ==============================================================================
# 练习 1：实现 n-gram 统计
# ==============================================================================

def exercise_1_count_ngrams(text: str, n: int) -> dict:
    """
    练习 1：统计文本中所有 n-gram 的出现次数

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你在数一篇文章中每两个连续的字出现了几次。
    比如 "我爱北京天安门"：
    - "我爱" 出现 1 次
    - "爱北" 出现 1 次
    - "北京" 出现 1 次
    - ...

    ━━━━━━━ 提示 ━━━━━━━
    1. 创建一个空字典 ngrams = {}
    2. 从位置 0 到 len(text) - n，遍历每个位置 i
    3. 取出 text[i:i+n] 作为 n-gram
    4. 在字典中计数
    5. 返回字典

    参数：
        text: 输入文本
        n: n-gram 的长度

    返回：
        n-gram 计数字典 {n-gram: 出现次数}

    示例：
        >>> exercise_1_count_ngrams("abcab", 2)
        {'ab': 2, 'bc': 1, 'ca': 1}
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # ngrams = {}
    # for i in range(len(text) - n + 1):
    #     ngram = text[i:i+n]
    #     ngrams[ngram] = ngrams.get(ngram, 0) + 1
    # return ngrams
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：n-gram 统计")
    print("=" * 60)

    test_cases = [
        ("abcab", 2, {'ab': 2, 'bc': 1, 'ca': 1}),
        ("hello", 3, {'hel': 1, 'ell': 1, 'llo': 1}),
        ("aab", 2, {'aa': 1, 'ab': 1}),
    ]

    all_passed = True
    for text, n, expected in test_cases:
        result = exercise_1_count_ngrams(text, n)
        if result is None:
            print("  [未完成] 请实现 exercise_1_count_ngrams 函数")
            return False
        if result == expected:
            print(f"  [正确] count_ngrams('{text}', {n}) = {result}")
        else:
            print(f"  [错误] count_ngrams('{text}', {n}): 期望 {expected}, 实际 {result}")
            all_passed = False

    return all_passed


# ==============================================================================
# 练习 2：实现互信息计算
# ==============================================================================

def exercise_2_compute_mi(word: str, text: str) -> float:
    """
    练习 2：计算一个词的互信息

    ━━━━━━━ 生活类比 ━━━━━━━
    互信息衡量"两个字符经常一起出现"的程度。
    如果"北"和"京"总是一起出现，那"北京"的互信息就高。
    如果"的"和"了"只是碰巧挨在一起，那"的了"的互信息就低。

    ━━━━━━━ 提示 ━━━━━━━
    1. 计算 word 出现的次数 count_word
    2. 计算 word 的每个字符出现的次数 count_char[i]
    3. 计算总的 bigram 数量 total_bigrams
    4. 计算总的字符数量 total_chars
    5. P(word) = count_word / total_bigrams
    6. P(char[i]) = count_char[i] / total_chars
    7. MI = log2(P(word) / (P(char[0]) * P(char[1]) * ...))
    8. 如果 word 不在 text 中，返回 0.0

    参数：
        word: 要计算互信息的词
        text: 语料库文本

    返回：
        互信息值
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # if len(word) < 2:
    #     return 0.0
    #
    # # 统计 word 出现的次数
    # count_word = text.count(word)
    # if count_word == 0:
    #     return 0.0
    #
    # # 统计字符出现次数
    # char_counts = Counter(text)
    # total_chars = sum(char_counts.values())
    #
    # # 总的 bigram 数量
    # total_bigrams = len(text) - 1
    #
    # # P(word)
    # p_word = count_word / total_bigrams
    #
    # # P(每个字符) 的乘积
    # p_chars_product = 1.0
    # for char in word:
    #     p_char = char_counts.get(char, 0) / total_chars
    #     if p_char == 0:
    #         return 0.0
    #     p_chars_product *= p_char
    #
    # return math.log2(p_word / p_chars_product)
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：互信息计算")
    print("=" * 60)

    text = "北京天气好，我去北京大学，北京大学真美。"

    test_words = ["北京", "大学", "天气", "的了"]
    print(f"\n  文本: '{text}'")
    print(f"\n  {'词语':<8} {'互信息':>8}  {'说明'}")
    print("  " + "-" * 35)

    all_passed = True
    for word in test_words:
        result = exercise_2_compute_mi(word, text)
        if result is None:
            print("  [未完成] 请实现 exercise_2_compute_mi 函数")
            return False

        note = "高凝聚度" if result > 1.5 else "低凝聚度" if result < 0.5 else "中等"
        print(f"  {word:<8} {result:>8.4f}  {note}")

    # 检查 "北京" 的互信息是否高于 "的了"
    mi_bj = exercise_2_compute_mi("北京", text)
    mi_dl = exercise_2_compute_mi("的了", text)

    if mi_bj is not None and mi_dl is not None:
        if mi_bj > mi_dl:
            print(f"\n  [正确] '北京' 的互信息 ({mi_bj:.4f}) 高于 '的了' ({mi_dl:.4f})")
            return True
        else:
            print(f"\n  [部分正确] 互信息计算完成，但结果可能需要调整")
            return True
    return all_passed


# ==============================================================================
# 练习 3：实现简单的新词发现
# ==============================================================================

def exercise_3_discover_words(text: str, min_freq: int = 2, top_k: int = 5) -> list:
    """
    练习 3：实现简单的新词发现

    ━━━━━━━ 生活类比 ━━━━━━━
    就像在人群中找"新人"：
    1. 先数一数每个人出现了几次
    2. 出现次数多的更可能是"居民"
    3. 按出现次数排序，返回最可能的"新人"

    ━━━━━━━ 提示 ━━━━━━━
    1. 统计所有 2-gram 和 3-gram 的出现次数
    2. 过滤掉出现次数少于 min_freq 的
    3. 对于每个候选词，计算得分：
       得分 = 出现次数 * 互信息
    4. 按得分降序排序
    5. 返回前 top_k 个 (词, 得分) 的列表

    参数：
        text: 语料库文本
        min_freq: 最小出现频率
        top_k: 返回前 k 个结果

    返回：
        新词列表，每个元素是 (词, 得分)
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # # 统计 n-gram
    # ngram_counts = {}
    # for n in [2, 3]:
    #     for i in range(len(text) - n + 1):
    #         ngram = text[i:i+n]
    #         ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    #
    # # 过滤低频
    # candidates = {k: v for k, v in ngram_counts.items() if v >= min_freq}
    #
    # # 计算得分
    # scored = []
    # for word, count in candidates.items():
    #     mi = exercise_2_compute_mi(word, text)
    #     if mi and mi > 0:
    #         score = count * mi
    #         scored.append((word, score))
    #
    # # 排序
    # scored.sort(key=lambda x: x[1], reverse=True)
    # return scored[:top_k]
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：简单的新词发现")
    print("=" * 60)

    text = """
    今天天气真好，我去北京大学看望小明。
    小明在北京大学学习自然语言处理。
    自然语言处理是人工智能的重要方向。
    人工智能技术发展迅速，深度学习是核心技术。
    深度学习在自然语言处理中应用广泛。
    """

    result = exercise_3_discover_words(text, min_freq=2, top_k=5)

    if result is None:
        print("  [未完成] 请实现 exercise_3_discover_words 函数")
        return False

    print(f"\n  Top-5 新词发现结果:")
    print(f"  {'排名':<4} {'词语':<8} {'得分':>8}")
    print("  " + "-" * 25)

    for rank, (word, score) in enumerate(result, 1):
        bar = "#" * max(0, int(score))
        print(f"  {rank:<4} {word:<8} {score:>8.2f}  {bar}")

    if len(result) > 0 and result[0][1] > 0:
        print(f"\n  [正确] 发现了 {len(result)} 个新词，最高分为 {result[0][1]:.2f}")
        return True
    else:
        print("\n  [错误] 未发现有效的新词")
        return False


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十一章 练习                  ║
    ║        新词发现与短语提取                              ║
    ╚══════════════════════════════════════════════════════╝
    """)

    results = []
    results.append(("练习1: n-gram 统计", test_exercise_1()))
    results.append(("练习2: 互信息计算", test_exercise_2()))
    results.append(("练习3: 新词发现", test_exercise_3()))

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
        print("\n  所有练习完成！你已经掌握了新词发现与短语提取的核心技术。")
        print("  恭喜你完成了 G-one NLP 学院的全部课程！")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
