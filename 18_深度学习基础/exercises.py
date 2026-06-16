import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十八章：深度学习基础 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 实现激活函数
    2. 实现感知机训练
    3. 实现前向传播

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


# ==============================================================================
# 练习 1：实现激活函数
# ==============================================================================

def exercise_1_activation(x: float, function_name: str) -> float:
    """
    练习 1：实现常见的激活函数

    ━━━━━━━ 生活类比 ━━━━━━━
    激活函数就像一个"转换器"：
    - Sigmoid：把任意数字压缩到 0~1 之间（像百分比）
    - ReLU：负数变 0，正数不变（像单向阀）
    - Tanh：把任意数字压缩到 -1~1 之间（像温度计）

    ━━━━━━━ 提示 ━━━━━━━
    1. Sigmoid: σ(x) = 1 / (1 + exp(-x))
       - 注意：当 x 很小时，exp(-x) 会溢出
       - 可以用条件判断处理

    2. ReLU: f(x) = max(0, x)

    3. Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
       - 或者直接用 math.tanh(x)

    参数：
        x: 输入值
        function_name: 激活函数名称 ("sigmoid", "relu", "tanh")

    返回：
        激活后的值

    示例：
        >>> exercise_1_activation(0, "sigmoid")
        0.5
        >>> exercise_1_activation(-1, "relu")
        0
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # if function_name == "sigmoid":
    #     if x >= 0:
    #         return 1.0 / (1.0 + math.exp(-x))
    #     else:
    #         return math.exp(x) / (1.0 + math.exp(x))
    # elif function_name == "relu":
    #     return max(0, x)
    # elif function_name == "tanh":
    #     return math.tanh(x)
    # else:
    #     raise ValueError(f"未知的激活函数: {function_name}")
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：实现激活函数")
    print("=" * 60)

    test_cases = [
        (0, "sigmoid", 0.5),
        (0, "relu", 0),
        (0, "tanh", 0),
        (1, "sigmoid", 0.7311),
        (-1, "relu", 0),
        (2, "relu", 2),
    ]

    all_correct = True
    for x, func, expected in test_cases:
        result = exercise_1_activation(x, func)
        if result is None:
            print(f"[未完成] 请实现 exercise_1_activation 函数")
            return False
        if abs(result - expected) < 0.01:
            print(f"  [正确] {func}({x}) = {result:.4f}")
        else:
            print(f"  [错误] {func}({x}) = {result:.4f}, 期望: {expected:.4f}")
            all_correct = False

    return all_correct


# ==============================================================================
# 练习 2：实现感知机训练
# ==============================================================================

def exercise_2_train_perceptron(training_data: list, labels: list,
                                 epochs: int = 100, learning_rate: float = 0.1) -> tuple:
    """
    练习 2：实现感知机的训练过程

    ━━━━━━━ 生活类比 ━━━━━━━
    就像训练一个评委：
    1. 评委看一个求职者，给出判断
    2. 和正确答案对比
    3. 如果判断错了，就调整评判标准
    4. 重复多次，直到判断准确

    ━━━━━━━ 提示 ━━━━━━━
    1. 初始化权重和偏置
       - weights = [0.0] * len(training_data[0])
       - bias = 0.0

    2. 循环 epochs 次：
       a. 对每个训练样本 (inputs, label)：
          - 计算输出：output = sigmoid(Σ(wi * xi) + bias)
          - 计算误差：error = label - output
          - 更新权重：wi += learning_rate * error * xi
          - 更新偏置：bias += learning_rate * error

    3. 返回 (weights, bias)

    参数：
        training_data: 训练数据（每行是一个样本的特征）
        labels: 标签
        epochs: 训练轮数
        learning_rate: 学习率

    返回：
        (weights, bias) 元组

    示例：
        >>> data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        >>> labels = [0, 0, 0, 1]
        >>> w, b = exercise_2_train_perceptron(data, labels)
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # n_inputs = len(training_data[0])
    # weights = [0.0] * n_inputs
    # bias = 0.0
    #
    # def sigmoid(x):
    #     if x >= 0:
    #         return 1.0 / (1.0 + math.exp(-x))
    #     else:
    #         return math.exp(x) / (1.0 + math.exp(x))
    #
    # for _ in range(epochs):
    #     for inputs, label in zip(training_data, labels):
    #         # 前向传播
    #         total = sum(w * x for w, x in zip(weights, inputs)) + bias
    #         output = sigmoid(total)
    #         # 计算误差
    #         error = label - output
    #         # 更新权重
    #         for i in range(n_inputs):
    #             weights[i] += learning_rate * error * inputs[i]
    #         # 更新偏置
    #         bias += learning_rate * error
    #
    # return weights, bias
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：感知机训练")
    print("=" * 60)

    # AND 逻辑
    and_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    and_labels = [0, 0, 0, 1]

    result = exercise_2_train_perceptron(and_data, and_labels, epochs=100, learning_rate=0.5)

    if result is None:
        print("[未完成] 请实现 exercise_2_train_perceptron 函数")
        return False

    weights, bias = result

    # 测试训练结果
    def sigmoid(x):
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            return math.exp(x) / (1.0 + math.exp(x))

    correct = 0
    for inputs, label in zip(and_data, and_labels):
        total = sum(w * x for w, x in zip(weights, inputs)) + bias
        output = sigmoid(total)
        predicted = 1 if output > 0.5 else 0
        if predicted == label:
            correct += 1
        print(f"    {inputs} → {output:.3f} (预测: {predicted}, 期望: {label})")

    if correct == 4:
        print(f"[正确] AND 逻辑学习成功！")
        return True
    else:
        print(f"[结果] 正确率: {correct}/4")
        return False


# ==============================================================================
# 练习 3：实现简单的前向传播
# ==============================================================================

def exercise_3_forward(inputs: list, weights: list, bias: list, activation: str) -> list:
    """
    练习 3：实现单层神经网络的前向传播

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一个"加权评分系统"：
    - 每个输入代表一个评委的分数
    - 每个权重代表这个评委的重要程度
    - 最终得分 = 评委分数 × 重要程度 之和

    ━━━━━━━ 提示 ━━━━━━━
    1. 对每个输出神经元 j：
       - 计算加权和：sum_j = Σ(inputs[i] * weights[i][j]) + bias[j]
       - 应用激活函数：output_j = activation(sum_j)

    2. 激活函数选择：
       - "sigmoid": 1 / (1 + exp(-x))
       - "relu": max(0, x)

    3. 返回输出列表

    参数：
        inputs: 输入列表 [x1, x2, ...]
        weights: 权重矩阵 [[w11, w12], [w21, w22], ...]
                 weights[i][j] 表示输入 i 到输出 j 的权重
        bias: 偏置列表 [b1, b2, ...]
        activation: 激活函数名称 ("sigmoid" 或 "relu")

    返回：
        输出列表

    示例：
        >>> inputs = [1.0, 0.5]
        >>> weights = [[0.5, 0.3], [0.2, 0.8]]
        >>> bias = [0.1, -0.1]
        >>> exercise_3_forward(inputs, weights, bias, "sigmoid")
        [0.69..., 0.62...]  （近似值）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # n_outputs = len(bias)
    # outputs = []
    #
    # for j in range(n_outputs):
    #     # 计算加权和
    #     total = sum(inputs[i] * weights[i][j] for i in range(len(inputs))) + bias[j]
    #     # 应用激活函数
    #     if activation == "sigmoid":
    #         if total >= 0:
    #             output = 1.0 / (1.0 + math.exp(-total))
    #         else:
    #             output = math.exp(total) / (1.0 + math.exp(total))
    #     elif activation == "relu":
    #         output = max(0, total)
    #     else:
    #         output = total
    #     outputs.append(output)
    #
    # return outputs
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：前向传播")
    print("=" * 60)

    inputs = [1.0, 0.5]
    weights = [[0.5, 0.3], [0.2, 0.8]]
    bias = [0.1, -0.1]

    result = exercise_3_forward(inputs, weights, bias, "sigmoid")

    if result is None:
        print("[未完成] 请实现 exercise_3_forward 函数")
        return False

    if not isinstance(result, list) or len(result) != 2:
        print(f"[错误] 返回值应该是长度为 2 的列表")
        return False

    # 手动计算期望值
    # 输出 0: sigmoid(1.0*0.5 + 0.5*0.2 + 0.1) = sigmoid(0.7) ≈ 0.668
    # 输出 1: sigmoid(1.0*0.3 + 0.5*0.8 - 0.1) = sigmoid(0.6) ≈ 0.646
    expected_0 = 1.0 / (1.0 + math.exp(-0.7))
    expected_1 = 1.0 / (1.0 + math.exp(-0.6))

    if abs(result[0] - expected_0) < 0.01 and abs(result[1] - expected_1) < 0.01:
        print(f"[正确] 输入: {inputs}")
        print(f"       输出: {[f'{x:.4f}' for x in result]}")
        print(f"       期望: {[f'{expected_0:.4f}', f'{expected_1:.4f}']}")
        return True
    else:
        print(f"[错误] 输入: {inputs}")
        print(f"       输出: {[f'{x:.4f}' for x in result]}")
        print(f"       期望: {[f'{expected_0:.4f}', f'{expected_1:.4f}']}")
        return False


# ==============================================================================
# 主程序：运行所有练习测试
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十八章 练习                  ║
    ║        深度学习基础                                  ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 运行所有练习测试
    results = []
    results.append(("练习1: 激活函数", test_exercise_1()))
    results.append(("练习2: 感知机训练", test_exercise_2()))
    results.append(("练习3: 前向传播", test_exercise_3()))

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
        print("  你已经掌握了深度学习的基础知识。")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
        print("  不要着急，慢慢来，理解了再写代码。")
