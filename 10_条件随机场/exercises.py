import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十章：条件随机场（CRF）— 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 实现中文特征提取
    2. 实现维特比算法
    3. 实现简单的序列标注器

运行方式：
    python exercises.py
==============================================================================
"""

import math


# ==============================================================================
# 练习 1：实现中文特征提取
# ==============================================================================

def exercise_1_extract_features(sentence: list, index: int) -> dict:
    """
    练习 1：提取中文句子中某个位置的特征

    ━━━━━━━ 生活类比 ━━━━━━━
    就像面试时，面试官会问你：
    - 你自己是谁？（当前词）
    - 你前面的人是谁？（前一个词）
    - 你后面的人是谁？（后一个词）

    ━━━━━━━ 提示 ━━━━━━━
    1. 获取当前字符 char = sentence[index]
    2. 添加特征：
       - 'char': 当前字符
       - 'char.isdigit': 是否是数字
       - 'char.is_chinese': 是否是中文（'一' <= char <= '鿿'）
    3. 如果 index > 0，添加前一个字符的特征：
       - '-1:char': 前一个字符
    4. 如果 index < len(sentence) - 1，添加后一个字符的特征：
       - '+1:char': 后一个字符
    5. 返回特征字典

    参数：
        sentence: 句子（字符列表）
        index: 当前字符的位置

    返回：
        特征字典
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # char = sentence[index]
    # features = {
    #     'char': char,
    #     'char.isdigit': char.isdigit(),
    #     'char.is_chinese': '一' <= char <= '鿿',
    # }
    # if index > 0:
    #     features['-1:char'] = sentence[index - 1]
    # if index < len(sentence) - 1:
    #     features['+1:char'] = sentence[index + 1]
    # return features
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：中文特征提取")
    print("=" * 60)

    sentence = ["小", "明", "在", "北", "京"]
    index = 2  # "在"

    result = exercise_1_extract_features(sentence, index)

    if result is None:
        print("  [未完成] 请实现 exercise_1_extract_features 函数")
        return False

    print(f"  句子: {sentence}")
    print(f"  位置 {index}: '{sentence[index]}'")
    print(f"  特征:")

    # 检查必需的特征
    required = ['char', 'char.isdigit', 'char.is_chinese', '-1:char', '+1:char']
    all_passed = True

    for key in required:
        if key in result:
            print(f"    {key}: {result[key]}")
        else:
            print(f"    [缺少] {key}")
            all_passed = False

    if all_passed:
        print("  [正确] 所有必需特征都存在")
    return all_passed


# ==============================================================================
# 练习 2：实现维特比算法
# ==============================================================================

def exercise_2_viterbi(scores_matrix: list, transition_matrix: list) -> list:
    """
    练习 2：实现维特比算法

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你在走一个迷宫：
    - 每一层有很多房间（可能的标签）
    - 每个房间有一个"得分"
    - 从一个房间走到下一个房间也有"得分"
    维特比算法就是找到一条"总得分最高"的路径。

    ━━━━━━━ 提示 ━━━━━━━
    1. scores_matrix[i][j] = 位置 i 的标签 j 的得分
    2. transition_matrix[j][k] = 从标签 j 转移到标签 k 的得分
    3. dp[i][j] = 到达位置 i 的标签 j 的最大得分
    4. backpointer[i][j] = 到达位置 i 的标签 j 的最优前驱
    5. 初始化：dp[0][j] = scores_matrix[0][j]
    6. 递推：dp[i][j] = max(dp[i-1][k] + transition[k][j] + scores[i][j])
    7. 回溯：从最后一个位置开始，找到最优路径

    参数：
        scores_matrix: 得分矩阵，形状为 (序列长度, 标签数)
        transition_matrix: 转移得分矩阵，形状为 (标签数, 标签数)

    返回：
        最优标签索引序列
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # n = len(scores_matrix)
    # num_labels = len(scores_matrix[0])
    # dp = [[-float('inf')] * num_labels for _ in range(n)]
    # backpointer = [[0] * num_labels for _ in range(n)]
    #
    # for j in range(num_labels):
    #     dp[0][j] = scores_matrix[0][j]
    #
    # for i in range(1, n):
    #     for j in range(num_labels):
    #         best_score = -float('inf')
    #         best_prev = 0
    #         for k in range(num_labels):
    #             score = dp[i-1][k] + transition_matrix[k][j] + scores_matrix[i][j]
    #             if score > best_score:
    #                 best_score = score
    #                 best_prev = k
    #         dp[i][j] = best_score
    #         backpointer[i][j] = best_prev
    #
    # path = [0] * n
    # path[-1] = max(range(num_labels), key=lambda j: dp[n-1][j])
    # for i in range(n - 2, -1, -1):
    #     path[i] = backpointer[i + 1][path[i + 1]]
    # return path
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：维特比算法")
    print("=" * 60)

    # 简单测试：3 个位置，2 个标签
    scores = [
        [1.0, 0.5],   # 位置 0：标签 0 得分 1.0，标签 1 得分 0.5
        [0.3, 0.8],   # 位置 1：标签 0 得分 0.3，标签 1 得分 0.8
        [0.6, 0.4],   # 位置 2：标签 0 得分 0.6，标签 1 得分 0.4
    ]

    # 转移矩阵：标签 0 → 标签 0 得分 0.5，标签 0 → 标签 1 得分 0.5
    transitions = [
        [0.5, 0.5],   # 从标签 0 转移
        [0.5, 0.5],   # 从标签 1 转移
    ]

    result = exercise_2_viterbi(scores, transitions)

    if result is None:
        print("  [未完成] 请实现 exercise_2_viterbi 函数")
        return False

    print(f"  得分矩阵: {scores}")
    print(f"  转移矩阵: {transitions}")
    print(f"  最优路径: {result}")

    # 计算路径得分
    total_score = scores[0][result[0]]
    for i in range(1, len(result)):
        total_score += transitions[result[i-1]][result[i]] + scores[i][result[i]]
    print(f"  路径得分: {total_score:.2f}")

    if len(result) == 3:
        print("  [正确] 返回了正确长度的路径")
        return True
    else:
        print("  [错误] 路径长度不正确")
        return False


# ==============================================================================
# 练习 3：实现简单的序列标注器
# ==============================================================================

def exercise_3_sequence_labeling(train_data: list, test_sentence: list) -> list:
    """
    练习 3：实现一个简单的序列标注器

    ━━━━━━━ 生活类比 ━━━━━━━
    就像教小朋友认字：
    - 先给他看一些例子（训练数据）
    - 然后让他认新的字（测试数据）

    ━━━━━━━ 提示 ━━━━━━━
    1. 统计每个字符对应的标签频率
       char_label_count = {char: {label: count}}
    2. 对于测试句子中的每个字符：
       - 如果在训练数据中见过，选择出现最多的标签
       - 如果没见过，选择整体最常见的标签
    3. 返回标签列表

    参数：
        train_data: 训练数据，格式为 [(句子, 标签), ...]
                   句子是字符列表，标签是字符串列表
        test_sentence: 测试句子（字符列表）

    返回：
        预测的标签列表
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # # 统计字符-标签频率
    # char_label_count = {}
    # label_count = {}
    # for sent, labels in train_data:
    #     for char, label in zip(sent, labels):
    #         if char not in char_label_count:
    #             char_label_count[char] = {}
    #         char_label_count[char][label] = char_label_count[char].get(label, 0) + 1
    #         label_count[label] = label_count.get(label, 0) + 1
    #
    # # 找最常见的标签
    # default_label = max(label_count, key=label_count.get)
    #
    # # 预测
    # result = []
    # for char in test_sentence:
    #     if char in char_label_count:
    #         best_label = max(char_label_count[char], key=char_label_count[char].get)
    #         result.append(best_label)
    #     else:
    #         result.append(default_label)
    # return result
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：简单序列标注器")
    print("=" * 60)

    train_data = [
        (["小", "明", "去", "北", "京"], ["B", "E", "O", "B", "E"]),
        (["小", "红", "在", "上", "海"], ["B", "E", "O", "B", "E"]),
        (["今", "天", "天", "气", "好"], ["O", "O", "O", "O", "O"]),
    ]

    test_sent = ["小", "华", "去", "上", "海"]

    result = exercise_3_sequence_labeling(train_data, test_sent)

    if result is None:
        print("  [未完成] 请实现 exercise_3_sequence_labeling 函数")
        return False

    print(f"  训练数据: {len(train_data)} 个样本")
    print(f"  测试句子: {test_sent}")
    print(f"  预测标签: {result}")

    # 检查结果
    if len(result) == len(test_sent):
        print("  [正确] 返回了正确长度的标签序列")
        # 检查 "小" 是否标为 "B"
        if result[0] == "B":
            print("  [正确] '小' 被正确标注为 'B'")
            return True
        else:
            print("  [部分正确] 标签长度正确，但部分标签可能需要调整")
            return True
    else:
        print("  [错误] 标签长度不正确")
        return False


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十章 练习                    ║
    ║        条件随机场（CRF）                              ║
    ╚══════════════════════════════════════════════════════╝
    """)

    results = []
    results.append(("练习1: 特征提取", test_exercise_1()))
    results.append(("练习2: 维特比算法", test_exercise_2()))
    results.append(("练习3: 序列标注器", test_exercise_3()))

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
        print("\n  所有练习完成！你已经掌握了条件随机场的核心技术。")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
