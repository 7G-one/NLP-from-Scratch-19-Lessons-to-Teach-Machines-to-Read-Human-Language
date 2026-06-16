"""
==============================================================================
第十章：条件随机场（CRF，Conditional Random Fields）
==============================================================================
日期：2026-05-16

同学们好！这节课我们来学习序列标注中的"终极武器" —— 条件随机场。

----------------------------------------------------------------------
生活类比：通过上下文判断一个词的"身份"
----------------------------------------------------------------------

想象你是一个侦探，面前有一排人，你需要判断每个人的职业。

如果你只看一个人的外表，很难判断。
但如果你看"上下文"，就容易多了：

  ... 医生 护士 [?] 病人 ...
  → 问号处很可能也是医院里的人（比如"药剂师"）

  ... 厨师 服务员 [?] 顾客 ...
  → 问号处很可能是餐厅里的人（比如"收银员"）

CRF 做的就是这件事：
  不是孤立地看每个词，而是综合考虑"前后文"来判断每个词的标签。

----------------------------------------------------------------------
CRF vs 其他模型
----------------------------------------------------------------------

1. HMM（隐马尔可夫模型）：
   - 假设：当前状态只依赖于前一个状态（马尔可夫假设）
   - 问题：假设太强，很多情况下不成立

2. 最大熵模型（MaxEnt）：
   - 只看当前词的特征，不考虑上下文
   - 问题：忽略了序列信息

3. CRF（条件随机场）：
   - 既看当前词的特征，也看上下文
   - 不需要独立性假设
   - 是序列标注中最强大的模型之一

----------------------------------------------------------------------
本章内容
----------------------------------------------------------------------

1. CRF 的基本概念
2. 特征模板设计
3. 简化的 CRF 实现
4. sklearn-crfsuite 使用

==============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import numpy as np


# ==============================================================================
# 第一部分：CRF 的基本概念
# ==============================================================================
#
# CRF 是一种判别式模型（Discriminative Model）。
#
# 什么是判别式模型？
#   直接学习 P(标签|输入)，而不是 P(输入, 标签)。
#
# 生活类比：
#   生成式模型（如 HMM）：学习"什么样的人会穿什么样的衣服"
#     → 需要了解所有可能的衣服和人的组合
#
#   判别式模型（如 CRF）：直接学习"看到这件衣服，判断穿的人是谁"
#     → 只关注如何区分，不需要了解所有组合
#
# CRF 的核心公式：
#   P(Y|X) = (1/Z) * exp(Σ λ_k * f_k(y_{i-1}, y_i, X, i))
#
# 其中：
#   - Y = 标签序列（我们要预测的）
#   - X = 输入序列（我们已知的）
#   - f_k = 特征函数（衡量某种模式是否存在）
#   - λ_k = 特征权重（通过训练学到的）
#   - Z = 归一化常数（确保概率之和为 1）
#
# ==============================================================================


def softmax(scores: list) -> list:
    """
    计算 softmax 函数（将分数转换为概率分布）

    ━━━━━━━ 生活类比 ━━━━━━━
    想象一场比赛，三个人的得分是 [3, 1, 2]。
    softmax 会把这些分数转换成"获胜概率"：
    [0.67, 0.09, 0.24]
    得分最高的人，获胜概率最大。

    参数：
        scores: 分数列表

    返回：
        概率分布（和为 1）
    """
    # 减去最大值防止数值溢出
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def log_sum_exp(scores: list) -> float:
    """
    计算 log(sum(exp(scores)))，防止数值溢出

    ━━━━━━━ 为什么要用这个？ ━━━━━━━
    在 CRF 的训练中，我们需要计算 log(Σ exp(score))。
    如果直接计算，exp(score) 可能非常大，导致数值溢出。
    log_sum_exp 通过数学技巧避免了这个问题。

    参数：
        scores: 分数列表

    返回：
        log(sum(exp(scores)))
    """
    max_score = max(scores)
    return max_score + math.log(sum(math.exp(s - max_score) for s in scores))


# ==============================================================================
# 第二部分：特征模板设计
# ==============================================================================
#
# 特征模板是 CRF 的核心。
#
# 什么是特征模板？
#   特征模板定义了"从数据中提取哪些特征"。
#
# 生活类比：
#   想象你在面试一个人，你的"面试问题清单"就是特征模板：
#   1. 你的学历是什么？（当前词的特征）
#   2. 你之前的工作是什么？（前一个词的特征）
#   3. 你的期望薪资是多少？（当前位置的特征）
#
#   CRF 的特征模板也是类似的：
#   1. 当前词是什么？（Unigram 特征）
#   2. 前一个词是什么？（Bigram 特征）
#   3. 当前词的前后缀是什么？（词形特征）
#
# ==============================================================================


def word_features(sentence: list, index: int) -> dict:
    """
    提取当前词的特征

    ━━━━━━━ 特征说明 ━━━━━━━
    1. 当前词本身（bias）
    2. 前一个词（如果存在）
    3. 后一个词（如果存在）
    4. 当前词的前缀（前 2 个字符）
    5. 当前词的后缀（后 2 个字符）
    6. 当前词的长度
    7. 是否是数字
    8. 是否是英文
    9. 是否是标点

    参数：
        sentence: 句子（词列表）
        index: 当前词的位置

    返回：
        特征字典
    """
    word = sentence[index]

    features = {
        # ===== 当前词的特征 =====
        'bias': 1.0,                              # 偏置项
        'word': word,                              # 当前词
        'word.lower()': word.lower(),              # 小写形式
        'word[-2:]': word[-2:],                    # 后缀（最后 2 个字符）
        'word[-3:]': word[-3:],                    # 后缀（最后 3 个字符）
        'word[:2]': word[:2],                      # 前缀（前 2 个字符）
        'word[:3]': word[:3],                      # 前缀（前 3 个字符）
        'word.len': len(word),                     # 词长度
        'word.isdigit': word.isdigit(),            # 是否是数字
        'word.isalpha': word.isalpha(),            # 是否是字母
        'word.isupper': word.isupper(),            # 是否全大写
        'word.istitle': word.istitle(),            # 是否首字母大写
    }

    # ===== 前一个词的特征 =====
    if index > 0:
        prev_word = sentence[index - 1]
        features.update({
            '-1:word': prev_word,
            '-1:word.lower()': prev_word.lower(),
            '-1:word[-2:]': prev_word[-2:],
            '-1:word.isdigit': prev_word.isdigit(),
        })
    else:
        # 如果是第一个词，标记为"句子开始"
        features['BOS'] = True  # Beginning Of Sentence

    # ===== 后一个词的特征 =====
    if index < len(sentence) - 1:
        next_word = sentence[index + 1]
        features.update({
            '+1:word': next_word,
            '+1:word.lower()': next_word.lower(),
            '+1:word[:2]': next_word[:2],
            '+1:word.isdigit': next_word.isdigit(),
        })
    else:
        # 如果是最后一个词，标记为"句子结束"
        features['EOS'] = True  # End Of Sentence

    return features


def word_features_chinese(sentence: list, index: int) -> dict:
    """
    针对中文的特征提取

    ━━━━━━━ 中文特征说明 ━━━━━━━
    1. 当前字符
    2. 前一个字符
    3. 后一个字符
    4. 字符类型（汉字、数字、英文、标点）
    5. 前后字符的组合

    参数：
        sentence: 句子（字符列表）
        index: 当前字符的位置

    返回：
        特征字典
    """
    char = sentence[index]

    features = {
        'bias': 1.0,
        'char': char,
        'char.isdigit': char.isdigit(),
        'char.isalpha': char.isalpha(),
        'char.is_chinese': '一' <= char <= '鿿',
    }

    # 前一个字符
    if index > 0:
        prev = sentence[index - 1]
        features['-1:char'] = prev
        features['-1:char.isdigit'] = prev.isdigit()
        features['-1:char.is_chinese'] = '一' <= prev <= '鿿'
        # 前后字符组合特征
        features['-1:char+char'] = prev + char
    else:
        features['BOS'] = True

    # 后一个字符
    if index < len(sentence) - 1:
        nxt = sentence[index + 1]
        features['+1:char'] = nxt
        features['+1:char.isdigit'] = nxt.isdigit()
        features['+1:char.is_chinese'] = '一' <= nxt <= '鿿'
        features['char+1:char'] = char + nxt
    else:
        features['EOS'] = True

    # 前前一个字符（更远的上下文）
    if index > 1:
        features['-2:char'] = sentence[index - 2]

    # 后后一个字符
    if index < len(sentence) - 2:
        features['+2:char'] = sentence[index + 2]

    return features


def extract_sentence_features(sentence: list, is_chinese: bool = False) -> list:
    """
    提取整个句子的特征

    参数：
        sentence: 句子
        is_chinese: 是否是中文

    返回：
        特征列表（每个词一个特征字典）
    """
    if is_chinese:
        return [word_features_chinese(sentence, i) for i in range(len(sentence))]
    else:
        return [word_features(sentence, i) for i in range(len(sentence))]


# ==============================================================================
# 第三部分：简化的 CRF 实现
# ==============================================================================
#
# 这里我们实现一个简化版的 CRF，帮助理解其核心原理。
#
# 真正的 CRF 需要复杂的训练算法（如 L-BFGS），
# 这里我们用一个简化版本来演示核心思想。
#
# 生活类比：
#   真正的 CRF 就像一个经验丰富的侦探，能从无数线索中找到关键信息。
#   我们的简化版就像一个新手侦探，只关注最明显的几个线索。
#
# ==============================================================================


class SimpleCRF:
    """
    简化的 CRF 实现

    这个类演示了 CRF 的核心原理：
    1. 特征提取
    2. 权重计算
    3. 维特比解码（找最优标签序列）
    """

    def __init__(self):
        """初始化"""
        self.weights = {}       # 特征权重
        self.labels = []        # 所有可能的标签
        self.transition = {}    # 转移概率（标签之间的转移）

    def fit(self, X_features: list, y_labels: list, learning_rate: float = 0.01,
            epochs: int = 100):
        """
        训练 CRF（简化版本：使用感知机算法）

        ━━━━━━━ 算法说明 ━━━━━━━
        感知机算法的核心思想：
        1. 对于每个训练样本，用当前权重预测标签
        2. 如果预测错误，调整权重：
           - 增加正确标签对应特征的权重
           - 减少错误标签对应特征的权重

        生活类比：
          就像教小朋友认动物：
          - 给他看一张猫的照片
          - 如果他说"狗"，你就告诉他"这是猫"
          - 下次他再看到类似的猫，就会更倾向于说"猫"

        参数：
            X_features: 特征列表（每个句子的特征）
            y_labels: 标签列表（每个句子的标签）
            learning_rate: 学习率
            epochs: 训练轮数
        """
        # 收集所有标签
        all_labels = set()
        for labels in y_labels:
            all_labels.update(labels)
        self.labels = sorted(all_labels)

        # 初始化转移权重
        for l1 in self.labels:
            for l2 in self.labels:
                self.transition[(l1, l2)] = 0.0

        # 训练循环
        for epoch in range(epochs):
            errors = 0

            for features, true_labels in zip(X_features, y_labels):
                # 用维特比算法预测标签
                predicted_labels = self.viterbi(features)

                # 如果预测错误，更新权重
                if predicted_labels != true_labels:
                    errors += 1

                    # 增加正确标签的特征权重
                    for i, (feat, true_label) in enumerate(zip(features, true_labels)):
                        for key, value in feat.items():
                            if isinstance(value, bool):
                                value = 1.0 if value else 0.0
                            if isinstance(value, (int, float)):
                                weight_key = (key, true_label)
                                self.weights[weight_key] = self.weights.get(weight_key, 0) + learning_rate * value

                    # 减少错误标签的特征权重
                    for i, (feat, pred_label) in enumerate(zip(features, predicted_labels)):
                        for key, value in feat.items():
                            if isinstance(value, bool):
                                value = 1.0 if value else 0.0
                            if isinstance(value, (int, float)):
                                weight_key = (key, pred_label)
                                self.weights[weight_key] = self.weights.get(weight_key, 0) - learning_rate * value

                    # 更新转移权重
                    for i in range(1, len(true_labels)):
                        self.transition[(true_labels[i-1], true_labels[i])] += learning_rate
                        self.transition[(predicted_labels[i-1], predicted_labels[i])] -= learning_rate

            if (epoch + 1) % 10 == 0:
                accuracy = 1 - errors / len(X_features) if X_features else 0
                print(f"    Epoch {epoch+1}/{epochs}, 准确率: {accuracy:.2%}")

    def _score(self, features: dict, label: str) -> float:
        """
        计算某个标签在给定特征下的得分

        参数：
            features: 特征字典
            label: 标签

        返回：
            得分
        """
        score = 0.0
        for key, value in features.items():
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                weight_key = (key, label)
                score += self.weights.get(weight_key, 0) * value
        return score

    def viterbi(self, features: list) -> list:
        """
        维特比算法：找到最优标签序列

        ━━━━━━━ 生活类比 ━━━━━━━
        想象你在走一个迷宫：
        - 每一层有很多房间（可能的标签）
        - 每个房间有一个"得分"
        - 从一个房间走到下一个房间也有"得分"（转移得分）

        维特比算法就是找到一条"总得分最高"的路径。

        ━━━━━━━ 算法步骤 ━━━━━━━
        1. 初始化：计算第一个位置每个标签的得分
        2. 递推：对于每个位置，计算"从上一个位置的每个标签转移到当前标签"的得分
        3. 回溯：从最后一个位置开始，找到最优路径

        参数：
            features: 句子的特征列表

        返回：
            最优标签序列
        """
        n = len(features)
        if n == 0:
            return []

        num_labels = len(self.labels)
        label_to_idx = {l: i for i, l in enumerate(self.labels)}

        # dp[i][j] = 到达位置 i 的标签 j 的最大得分
        dp = [[-float('inf')] * num_labels for _ in range(n)]

        # backpointer[i][j] = 到达位置 i 的标签 j 的最优前驱标签
        backpointer = [[0] * num_labels for _ in range(n)]

        # 初始化第一个位置
        for j, label in enumerate(self.labels):
            dp[0][j] = self._score(features[0], label)

        # 递推
        for i in range(1, n):
            for j, curr_label in enumerate(self.labels):
                best_score = -float('inf')
                best_prev = 0

                for k, prev_label in enumerate(self.labels):
                    # 得分 = 前一个位置的得分 + 转移得分 + 当前特征得分
                    score = (dp[i-1][k] +
                            self.transition.get((prev_label, curr_label), 0) +
                            self._score(features[i], curr_label))

                    if score > best_score:
                        best_score = score
                        best_prev = k

                dp[i][j] = best_score
                backpointer[i][j] = best_prev

        # 回溯：找到最优路径
        path = [0] * n
        # 找到最后一个位置得分最高的标签
        path[-1] = max(range(num_labels), key=lambda j: dp[n-1][j])

        # 从后往前回溯
        for i in range(n - 2, -1, -1):
            path[i] = backpointer[i + 1][path[i + 1]]

        # 将索引转换为标签
        return [self.labels[idx] for idx in path]

    def predict(self, X_features: list) -> list:
        """
        预测标签序列

        参数：
            X_features: 特征列表

        返回：
            预测的标签序列
        """
        return [self.viterbi(features) for features in X_features]


# ==============================================================================
# 第四部分：sklearn-crfsuite 使用
# ==============================================================================
#
# 在实际项目中，我们通常使用 sklearn-crfsuite。
#
# sklearn-crfsuite 是一个基于 CRFsuite 的 Python 库，
# 它提供了简单易用的接口来训练和使用 CRF 模型。
#
# 安装方式：pip install sklearn-crfsuite
#
# ==============================================================================


def demo_sklearn_crfsuite():
    """
    演示 sklearn-crfsuite 的使用

    ━━━━━━━ 应用场景 ━━━━━━━
    命名实体识别（NER）：
    输入：今天 天气 很 好
    标签：O    O   O  O

    输入：小明 在 北京 大学 学习
    标签：B-PER O B-ORG I-ORG O
    """
    try:
        import sklearn_crfsuite
        from sklearn_crfsuite import metrics
    except ImportError:
        print("  [提示] sklearn-crfsuite 未安装")
        print("  请运行: pip install sklearn-crfsuite")
        return

    # 训练数据（简单的 NER 标注）
    train_sentences = [
        # (句子, 标签)
        (["小明", "在", "北京", "大学", "学习"], ["B-PER", "O", "B-ORG", "I-ORG", "O"]),
        (["小红", "喜欢", "上海"], ["B-PER", "O", "B-LOC"]),
        (["今天", "天气", "很", "好"], ["O", "O", "O", "O"]),
        (["苹果", "公司", "发布", "新", "手机"], ["B-ORG", "I-ORG", "O", "O", "O"]),
        (["李明", "去", "清华大学"], ["B-PER", "O", "B-ORG"]),
    ]

    # 提取特征
    X_train = [extract_sentence_features(s, is_chinese=True) for s, _ in train_sentences]
    y_train = [labels for _, labels in train_sentences]

    # 训练模型
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',       # 优化算法
        c1=0.1,                  # L1 正则化系数
        c2=0.1,                  # L2 正则化系数
        max_iterations=100,      # 最大迭代次数
        all_possible_transitions=True,  # 允许所有标签转移
    )

    crf.fit(X_train, y_train)

    # 预测
    test_sentences = [
        ["小明", "在", "清华大学", "学习"],
        ["苹果", "公司", "很", "大"],
    ]

    X_test = [extract_sentence_features(s, is_chinese=True) for s in test_sentences]
    y_pred = crf.predict(X_test)

    print("\n  预测结果:")
    for sent, labels in zip(test_sentences, y_pred):
        print(f"    句子: {sent}")
        print(f"    标签: {labels}")
        print()


def demo_crf_feature_importance():
    """
    演示 CRF 特征重要性分析

    ━━━━━━━ 有什么用？ ━━━━━━━
    通过分析特征权重，我们可以理解 CRF 模型"学到了什么"。
    比如，模型可能学到：
    - "大学"后面跟的词更可能是 "I-ORG"（机构名的一部分）
    - "小" 开头的词更可能是人名
    """
    try:
        import sklearn_crfsuite
    except ImportError:
        print("  [提示] sklearn-crfsuite 未安装")
        return

    # 更多训练数据
    train_data = [
        (["小明", "去", "北京大学"], ["B-PER", "O", "B-ORG"]),
        (["小红", "在", "清华大学"], ["B-PER", "O", "B-ORG"]),
        (["李华", "喜欢", "上海"], ["B-PER", "O", "B-LOC"]),
        (["王芳", "来自", "广东"], ["B-PER", "O", "B-LOC"]),
        (["今天", "天气", "好"], ["O", "O", "O"]),
        (["明天", "会", "下雨"], ["O", "O", "O"]),
    ]

    X = [extract_sentence_features(s, is_chinese=True) for s, _ in train_data]
    y = [labels for _, labels in train_data]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=50,
    )
    crf.fit(X, y)

    # 打印状态特征权重
    print("\n  CRF 学到的特征权重（前 10 个）:")
    from collections import Counter
    counter = Counter()
    for (attr, label), weight in crf.state_features_.items():
        counter[(attr, label)] = weight

    for (attr, label), weight in counter.most_common(10):
        print(f"    ({attr}, {label}): {weight:.4f}")


# ==============================================================================
# 第五部分：辅助工具
# ==============================================================================


def visualize_crf_transition(transition_probs: dict, labels: list):
    """
    可视化 CRF 转移概率

    参数：
        transition_probs: 转移概率字典 {(label1, label2): prob}
        labels: 标签列表
    """
    print("\n  转移概率矩阵:")
    print(f"  {'':>8}", end="")
    for l in labels:
        print(f"{l:>8}", end="")
    print()

    for l1 in labels:
        print(f"  {l1:>8}", end="")
        for l2 in labels:
            prob = transition_probs.get((l1, l2), 0)
            print(f"{prob:>8.3f}", end="")
        print()


def explain_crf_vs_hmm():
    """
    对比 CRF 和 HMM 的区别
    """
    print("""
    CRF vs HMM 对比：

    | 特性         | HMM                      | CRF                      |
    |-------------|--------------------------|--------------------------|
    | 模型类型     | 生成式模型                 | 判别式模型                |
    | 建模对象     | P(X, Y) 联合概率          | P(Y|X) 条件概率           |
    | 独立性假设   | 需要（马尔可夫假设）        | 不需要                    |
    | 特征         | 受限于状态转移和发射概率    | 可以使用任意特征           |
    | 训练         | EM 算法                   | 梯度下降 / L-BFGS         |
    | 推断         | 前向后向算法 / 维特比       | 维特比算法                |
    | 优势         | 简单快速                   | 灵活强大                  |
    | 劣势         | 特征受限                   | 训练较慢                  |

    在实际应用中，CRF 通常比 HMM 效果更好，
    因为它可以利用更丰富的特征。
    """)


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十章                        ║
    ║        条件随机场（CRF）                              ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 测试特征提取
    print("=" * 60)
    print("测试：特征提取")
    print("=" * 60)

    sentence = ["小明", "在", "北京", "大学", "学习"]
    features = extract_sentence_features(sentence, is_chinese=True)

    for i, (word, feat) in enumerate(zip(sentence, features)):
        print(f"\n  位置 {i}: '{word}'")
        for key, value in list(feat.items())[:5]:
            print(f"    {key}: {value}")

    # 测试简化 CRF
    print("\n" + "=" * 60)
    print("测试：简化 CRF")
    print("=" * 60)

    train_data = [
        (["我", "爱", "北京"], ["O", "O", "B-LOC"]),
        (["他", "去", "上海"], ["O", "O", "B-LOC"]),
        (["今天", "天气", "好"], ["O", "O", "O"]),
    ]

    X_train = [extract_sentence_features(s, is_chinese=True) for s, _ in train_data]
    y_train = [labels for _, labels in train_data]

    crf = SimpleCRF()
    crf.fit(X_train, y_train, epochs=20)

    test_sent = ["我", "爱", "上海"]
    test_feat = extract_sentence_features(test_sent, is_chinese=True)
    pred = crf.viterbi(test_feat)
    print(f"\n  测试: {test_sent}")
    print(f"  预测: {pred}")


# =============================================
# 课程总结
# =============================================
"""
核心收获：
- CRF 是判别式模型，直接学习 P(标签|输入)，不需要像 HMM 那样建模联合概率
- 特征模板是 CRF 的核心，包括当前词、前后词、字符类型、组合特征等
- 维特比算法通过动态规划找到全局最优的标签序列，而非贪心地逐个决策

常见陷阱：
- 特征模板设计太简单（只用当前词）会导致模型无法利用上下文信息
- 训练数据太少时 CRF 容易过拟合，需要配合正则化（L1/L2）使用
- 混淆 CRF 和 HMM 的适用场景——CRF 特征更灵活但训练更慢，HMM 简单快速但特征受限

下节课预告：
- 我们将学习新词发现与短语提取，用互信息和左右熵从文本中自动挖掘新词
"""
