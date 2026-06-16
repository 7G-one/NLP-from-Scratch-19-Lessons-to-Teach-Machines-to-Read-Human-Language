import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十七章：语言模型 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 实现简单的 Bigram 概率计算
    2. 实现文本生成
    3. 实现困惑度计算

运行方式：
    python exercises.py

提示：
    - 每个练习都有详细的提示，按照提示一步步来
    - 先自己写，写不出来再看注释中的参考答案
    - 运行后会自动检查你的答案是否正确
==============================================================================
"""

import math
import random
from collections import Counter


# ==============================================================================
# 练习 1：计算 Bigram 概率
# ==============================================================================

def exercise_1_bigram_probability(texts: list, prefix: str, word: str) -> float:
    """
    练习 1：计算 Bigram 条件概率 P(word | prefix)

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你是一个老师，记录了学生们的作文：
    - "今天" 出现了 100 次
    - "今天天气" 出现了 60 次
    - "今天好" 出现了 10 次

    那么 P("气" | "今天") = 60/100 = 0.6

    ━━━━━━━ 提示 ━━━━━━━
    1. 统计所有 Bigram（相邻的两个字符）
       - 遍历每个文本中的每一对相邻字符
       - bigram_counts[(c1, c2)] += 1
       - prefix_counts[c1] += 1

    2. 计算概率
       - P(word | prefix) = count(prefix, word) / count(prefix)

    3. 使用 Laplace 平滑（+1）
       - P = (count + 1) / (total + vocab_size)

    参数：
        texts: 训练文本列表
        prefix: 前缀字符（单个字符）
        word: 目标字符

    返回：
        条件概率（0 到 1 之间的浮点数）

    示例：
        >>> texts = ["今天天气好", "今天不错"]
        >>> exercise_1_bigram_probability(texts, "今", "天")
        1.0  （"今"后面总是"天"）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # # 收集所有字符构建词汇表
    # vocab = set()
    # bigram_counts = Counter()
    # prefix_counts = Counter()
    #
    # for text in texts:
    #     chars = [c for c in text if c.strip()]
    #     for i in range(len(chars) - 1):
    #         bigram_counts[(chars[i], chars[i+1])] += 1
    #         prefix_counts[chars[i]] += 1
    #         vocab.add(chars[i])
    #         vocab.add(chars[i+1])
    #
    # vocab_size = len(vocab)
    # count = bigram_counts.get((prefix, word), 0)
    # total = prefix_counts.get(prefix, 0)
    #
    # # Laplace 平滑
    # return (count + 1) / (total + vocab_size)
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：Bigram 概率计算")
    print("=" * 60)

    texts = ["今天天气好", "今天天气不错", "明天天气也好"]

    # 测试：P("天" | "今") 应该很高（"今"后面总是"天"）
    result = exercise_1_bigram_probability(texts, "今", "天")

    if result is None:
        print("[未完成] 请实现 exercise_1_bigram_probability 函数")
        return False

    if not isinstance(result, (int, float)):
        print(f"[错误] 返回值应该是数字")
        return False

    # "今"后面总是"天"，所以概率应该很高
    if result > 0.5:
        print(f"[正确] P('天' | '今') = {result:.4f}")
        print(f"       '今'后面总是跟'天'，概率很高 ✓")

        # 测试其他情况
        result2 = exercise_1_bigram_probability(texts, "好", "天")
        print(f"       P('天' | '好') = {result2:.4f}（应该较低）")
        return True
    else:
        print(f"[错误] P('天' | '今') = {result:.4f}")
        print(f"       期望 > 0.5（'今'后面总是跟'天'）")
        return False


# ==============================================================================
# 练习 2：简单的文本生成
# ==============================================================================

def exercise_2_generate_text(texts: list, seed: str, max_length: int = 5) -> str:
    """
    练习 2：使用 Bigram 模型生成文本

    ━━━━━━━ 生活类比 ━━━━━━━
    就像输入法的联想功能：
    1. 你输入一个字（种子）
    2. 输入法根据概率猜下一个字
    3. 把猜到的字加到后面
    4. 继续猜，直到达到指定长度

    ━━━━━━━ 提示 ━━━━━━━
    1. 构建 Bigram 模型
       - 统计每个字符后面跟哪些字符，以及各自的概率

    2. 从种子开始生成
       - 当前字符 = seed
       - 循环 max_length 次：
         a. 找到当前字符后面可能出现的所有字符
         b. 按概率随机选择一个
         c. 把选中的字符加到结果中
         d. 更新当前字符

    3. 返回生成的文本

    参数：
        texts: 训练文本列表
        seed: 种子字符
        max_length: 生成的最大长度

    返回：
        生成的文本（包含种子字符）

    示例：
        >>> texts = ["今天天气好", "今天不错"]
        >>> exercise_2_generate_text(texts, "今", 3)
        "今天气"  （可能的结果）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # # 构建 Bigram 模型
    # next_chars = {}  # {char: [next_char1, next_char2, ...]}
    # for text in texts:
    #     chars = [c for c in text if c.strip()]
    #     for i in range(len(chars) - 1):
    #         if chars[i] not in next_chars:
    #             next_chars[chars[i]] = []
    #         next_chars[chars[i]].append(chars[i+1])
    #
    # # 生成文本
    # result = seed
    # current = seed
    # for _ in range(max_length - 1):
    #     candidates = next_chars.get(current, [])
    #     if not candidates:
    #         break
    #     next_char = random.choice(candidates)
    #     result += next_char
    #     current = next_char
    #
    # return result
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：文本生成")
    print("=" * 60)

    texts = ["今天天气真好", "今天天气不错", "明天天气也好"]

    random.seed(42)  # 固定随机种子，保证结果可重复
    result = exercise_2_generate_text(texts, "今", max_length=4)

    if result is None:
        print("[未完成] 请实现 exercise_2_generate_text 函数")
        return False

    if not isinstance(result, str):
        print(f"[错误] 返回值应该是字符串")
        return False

    # 检查：应该以种子字符开头
    if not result.startswith("今"):
        print(f"[错误] 生成的文本应该以种子字符'今'开头")
        return False

    # 检查：长度应该 <= max_length
    if len(result) > 4:
        print(f"[错误] 生成的文本长度应该 <= {4}，实际: {len(result)}")
        return False

    print(f"[正确] 种子: '今', 最大长度: 4")
    print(f"       生成: '{result}'")
    return True


# ==============================================================================
# 练习 3：计算困惑度
# ==============================================================================

def exercise_3_perplexity(texts: list, test_text: str) -> float:
    """
    练习 3：计算语言模型的困惑度（Perplexity）

    ━━━━━━━ 生活类比 ━━━━━━━
    困惑度就像"考试分数"（但越低越好）：
    - 模型对训练数据中常见的文本不困惑 → PPL 低
    - 模型对没见过的文本很困惑 → PPL 高

    ━━━━━━━ 提示 ━━━━━━━
    1. 用训练文本构建 Bigram 模型
       - 统计每个字符后面跟哪些字符
       - 计算概率（使用 Laplace 平滑）

    2. 计算测试文本的困惑度
       - 对测试文本中的每一对相邻字符 (ci, ci+1)
       - 计算 P(ci+1 | ci)
       - 累加 log 概率：total_log_prob += log2(P)

    3. 计算困惑度
       - avg_log_prob = total_log_prob / (n - 1)  （n 是字符数）
       - PPL = 2^(-avg_log_prob)

    参数：
        texts: 训练文本列表
        test_text: 测试文本

    返回：
        困惑度（越低越好）

    示例：
        >>> texts = ["今天天气好", "今天不错"]
        >>> exercise_3_perplexity(texts, "今天天气好")
        低困惑度（和训练数据相似）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # # 构建词汇表和 Bigram 模型
    # vocab = set()
    # bigram_counts = Counter()
    # prefix_counts = Counter()
    #
    # for text in texts:
    #     chars = [c for c in text if c.strip()]
    #     for i in range(len(chars) - 1):
    #         bigram_counts[(chars[i], chars[i+1])] += 1
    #         prefix_counts[chars[i]] += 1
    #         vocab.add(chars[i])
    #         vocab.add(chars[i+1])
    #
    # vocab_size = len(vocab)
    #
    # # 计算困惑度
    # test_chars = [c for c in test_text if c.strip()]
    # total_log_prob = 0
    # n_pairs = 0
    #
    # for i in range(len(test_chars) - 1):
    #     count = bigram_counts.get((test_chars[i], test_chars[i+1]), 0)
    #     total = prefix_counts.get(test_chars[i], 0)
    #     prob = (count + 1) / (total + vocab_size)  # Laplace 平滑
    #     total_log_prob += math.log2(prob)
    #     n_pairs += 1
    #
    # if n_pairs == 0:
    #     return float('inf')
    #
    # avg_log_prob = total_log_prob / n_pairs
    # return 2 ** (-avg_log_prob)
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：困惑度计算")
    print("=" * 60)

    texts = ["今天天气好", "今天天气不错", "明天天气也好"]

    # 测试 1：和训练数据相似的文本
    result1 = exercise_3_perplexity(texts, "今天天气好")

    if result1 is None:
        print("[未完成] 请实现 exercise_3_perplexity 函数")
        return False

    if not isinstance(result1, (int, float)):
        print(f"[错误] 返回值应该是数字")
        return False

    # 测试 2：和训练数据不同的文本
    result2 = exercise_3_perplexity(texts, "苹果手机好")

    print(f"[正确] 测试结果:")
    print(f"       '今天天气好' (相似) → PPL = {result1:.2f}")
    print(f"       '苹果手机好' (不同) → PPL = {result2:.2f}")

    if result2 > result1:
        print(f"       不同文本的困惑度更高 ✓")
    else:
        print(f"       注意：不同文本的困惑度应该更高")

    return True


# ==============================================================================
# 主程序：运行所有练习测试
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十七章 练习                  ║
    ║        语言模型                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 运行所有练习测试
    results = []
    results.append(("练习1: Bigram 概率计算", test_exercise_1()))
    results.append(("练习2: 文本生成", test_exercise_2()))
    results.append(("练习3: 困惑度计算", test_exercise_3()))

    # 练习清单
    print("\n" + "=" * 60)
    print("  练习清单")
    print("=" * 60)
    for name, passed in results:
        status = "[完成]" if passed else "[未完成]"
        print(f"  {status} {name}")

    # 计算完成率
    completed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  完成率: {completed}/{total}")

    if completed == total:
        print("\n  恭喜！所有练习都完成了！")
        print("  你已经掌握了语言模型的核心概念。")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
        print("  不要着急，慢慢来，理解了再写代码。")
