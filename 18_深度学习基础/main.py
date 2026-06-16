import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十八章：深度学习基础 — 完整演示
==============================================================================
G-one NLP 学院
日期：2026-05-16

运行方式：
    python main.py

前置知识：
    - Python 基础（列表、函数、类）
    - 基本的数学概念（矩阵、导数）

本章内容：
    1. 感知机与逻辑门
    2. 激活函数对比
    3. 神经网络前向传播
    4. CNN（numpy 实现）
    5. RNN（numpy 实现）
    6. LSTM（numpy 实现）
    7. CNN/RNN/LSTM 概念
    8. PyTorch 实战
==============================================================================
"""

try:
    import numpy as np
except ImportError:
    print("本课需要 numpy 库，请运行：pip install numpy")
    print("安装后重新运行本文件即可。")
    exit(1)

from deep_learning import (
    Perceptron,
    SimpleNeuralNetwork,
    SimpleCNN,
    SimpleRNN,
    SimpleLSTM,
    sigmoid, relu, tanh, softmax,
    sigmoid_derivative, relu_derivative, tanh_derivative,
    cnn_concept,
    rnn_concept,
    lstm_concept,
    try_pytorch_demo,
)


def print_separator(title: str):
    """打印分隔线和标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def lesson_perceptron():
    """第一部分：感知机"""

    print_separator("18.1 感知机（Perceptron）")

    print("""
    感知机是最简单的神经网络：

    ┌─────────────────────────────────────────────────────────┐
    │   输入 → [权重] → 求和 → [激活函数] → 输出              │
    │                                                         │
    │   数学公式：output = σ(w1*x1 + w2*x2 + bias)            │
    │                                                         │
    │   感知机可以学习逻辑运算：                               │
    │   - AND（与）：两个都是 1 才输出 1                       │
    │   - OR（或）：至少一个是 1 就输出 1                      │
    │   - NOT（非）：取反                                      │
    └─────────────────────────────────────────────────────────┘
    """)

    # AND 逻辑
    print("  [实验 1] 学习 AND 逻辑:")
    and_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    and_labels = [0, 0, 0, 1]

    p_and = Perceptron(2)
    p_and.train(and_data, and_labels, epochs=100, learning_rate=0.5)

    print("    结果:")
    for inputs, label in zip(and_data, and_labels):
        output = p_and.forward(inputs)
        print(f"    {inputs} → {output:.3f} (期望: {label})")

    # OR 逻辑
    print("\n  [实验 2] 学习 OR 逻辑:")
    or_labels = [0, 1, 1, 1]

    p_or = Perceptron(2)
    p_or.train(and_data, or_labels, epochs=100, learning_rate=0.5)

    print("    结果:")
    for inputs, label in zip(and_data, or_labels):
        output = p_or.forward(inputs)
        print(f"    {inputs} → {output:.3f} (期望: {label})")


def lesson_activation_functions():
    """第二部分：激活函数"""

    print_separator("18.2 激活函数")

    print("""
    激活函数让神经网络具有"非线性"能力：

    ┌──────────┬────────────┬─────────────────────────────────┐
    │  函数     │  输出范围   │  特点                           │
    ├──────────┼────────────┼─────────────────────────────────┤
    │  Sigmoid │  (0, 1)    │  平滑，但有梯度消失问题          │
    │  ReLU    │  [0, +∞)   │  简单高效，最常用                │
    │  Tanh    │  (-1, 1)   │  零中心，比 Sigmoid 好           │
    │  Softmax │  (0, 1)    │  输出概率分布，用于多分类        │
    └──────────┴────────────┴─────────────────────────────────┘
    """)

    # 激活函数对比
    test_values = [-3, -2, -1, 0, 1, 2, 3]
    print(f"  {'输入':<6} {'Sigmoid':<10} {'ReLU':<8} {'Tanh':<10}")
    print(f"  {'-'*34}")
    for x in test_values:
        print(f"  {x:<6} {sigmoid(x):<10.4f} {relu(x):<8.1f} {tanh(x):<10.4f}")

    # Softmax 示例
    print(f"\n  Softmax 示例:")
    scores = [2.0, 1.0, 0.1]
    probs = softmax(scores)
    print(f"    原始分数: {scores}")
    print(f"    概率分布: {[f'{p:.3f}' for p in probs]}")
    print(f"    总和: {sum(probs):.3f}")


def lesson_forward_propagation():
    """第三部分：前向传播"""

    print_separator("18.3 前向传播")

    print("""
    前向传播就是数据从输入到输出的"流动过程"：

    ┌─────────────────────────────────────────────────────────┐
    │  输入层 → 隐藏层 1 → 隐藏层 2 → ... → 输出层            │
    │                                                         │
    │  每一层的计算：                                          │
    │  output = activation(weights × input + bias)            │
    │                                                         │
    │  就像流水线：                                           │
    │  原材料 → 加工1 → 加工2 → ... → 成品                   │
    └─────────────────────────────────────────────────────────┘
    """)

    # 创建神经网络
    model = SimpleNeuralNetwork(4, 8, 3)

    # 测试前向传播
    inputs = [1.0, 0.5, -0.5, 0.2]
    hidden, output = model.forward(inputs)

    print(f"  网络结构: 输入层(4) → 隐藏层(8) → 输出层(3)")
    print(f"  输入: {inputs}")
    print(f"  隐藏层输出（前 4 个）: {[f'{h:.3f}' for h in hidden[:4]]}")
    print(f"  最终输出: {[f'{o:.3f}' for o in output]}")
    print(f"  预测类别: {output.index(max(output))}")


def lesson_cnn_rnn_lstm():
    """第四部分：CNN/RNN/LSTM"""

    print_separator("18.4 CNN / RNN / LSTM")

    print("  CNN（卷积神经网络）:")
    cnn_concept()

    print("  RNN（循环神经网络）:")
    rnn_concept()

    print("  LSTM（长短期记忆网络）:")
    lstm_concept()


def lesson_numpy_cnn():
    """第四部分（上）：numpy 实现的 CNN"""

    print_separator("18.4a CNN —— numpy 从零实现")

    print("""
    CNN 的核心思想：用"放大镜"扫描文本，提取局部特征

    ┌─────────────────────────────────────────────────────────┐
    │  流程：词嵌入 → 卷积 → ReLU → 池化 → 全连接 → Softmax   │
    │                                                         │
    │  卷积核就像放大镜：                                     │
    │  "我 喜欢 自然 语言 处理"                               │
    │   [窗口1] → "我 喜欢" → 特征值                         │
    │    [窗口2] → "喜欢 自然" → 特征值                      │
    │     [窗口3] → "自然 语言" → 特征值                     │
    │      [窗口4] → "语言 处理" → 特征值                    │
    └─────────────────────────────────────────────────────────┘
    """)

    # ━━━━━ 创建 CNN 模型 ━━━━━
    vocab_size = 20       # 词汇表大小
    embed_dim = 8         # 词嵌入维度
    num_filters = 4       # 卷积核数量（4 种不同的"放大镜"）
    filter_size = 2       # 卷积核大小（每次看 2 个相邻的词）

    model = SimpleCNN(vocab_size, embed_dim, num_filters, filter_size)

    print(f"  模型参数:")
    print(f"    词汇表大小: {vocab_size}")
    print(f"    词嵌入维度: {embed_dim}")
    print(f"    卷积核数量: {num_filters}")
    print(f"    卷积核大小: {filter_size}")

    # ━━━━━ 模拟一个句子 ━━━━━
    # 假设句子是 "我(0) 喜欢(1) 这(2) 电影(3)"
    word_indices = np.array([0, 1, 2, 3])

    print(f"\n  输入句子（词索引）: {word_indices}")

    # 查看词嵌入
    embedded = model.embedding[word_indices]
    print(f"  词嵌入形状: {embedded.shape}")
    print(f"  第一个词的向量（前 4 维）: {np.array2string(embedded[0, :4], precision=3)}")

    # 查看卷积结果
    feature_maps = model.conv1d(embedded, model.filters)
    print(f"\n  卷积后的特征图形状: {feature_maps.shape}")
    print(f"  特征图（每个卷积核在每个位置的得分）:")
    for f_idx in range(num_filters):
        scores = np.array2string(feature_maps[f_idx], precision=3)
        print(f"    卷积核 {f_idx}: {scores}")

    # ReLU 激活
    feature_maps_relu = np.maximum(0, feature_maps)
    print(f"\n  ReLU 后（去掉负值）:")
    for f_idx in range(num_filters):
        scores = np.array2string(feature_maps_relu[f_idx], precision=3)
        print(f"    卷积核 {f_idx}: {scores}")

    # 最大池化
    pooled = model.max_pool(feature_maps_relu)
    print(f"\n  最大池化后（每个卷积核只保留最大值）:")
    print(f"    {np.array2string(pooled, precision=3)}")

    # 完整前向传播
    probs = model.forward(word_indices)
    print(f"\n  最终分类概率:")
    print(f"    正面: {probs[0]:.3f}  负面: {probs[1]:.3f}")
    print(f"    预测类别: {'正面' if probs[0] > probs[1] else '负面'}")


def lesson_numpy_rnn():
    """第四部分（中）：numpy 实现的 RNN"""

    print_separator("18.4b RNN —— numpy 从零实现")

    print("""
    RNN 的核心思想：有记忆的翻译官，逐词处理并保持记忆

    ┌─────────────────────────────────────────────────────────┐
    │  公式：h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)    │
    │                                                         │
    │  时间步 1        时间步 2        时间步 3                │
    │  ┌──────┐       ┌──────┐       ┌──────┐               │
    │  │ x(1) │       │ x(2) │       │ x(3) │  输入          │
    │  └──┬───┘       └──┬───┘       └──┬───┘               │
    │     ↓              ↓              ↓                     │
    │  ┌──────┐  ──→  ┌──────┐  ──→  ┌──────┐              │
    │  │ h(1) │       │ h(2) │       │ h(3) │  记忆传递      │
    │  └──┬───┘       └──┬───┘       └──┬───┘               │
    │     ↓              ↓              ↓                     │
    │  ┌──────┐       ┌──────┐       ┌──────┐              │
    │  │ y(1) │       │ y(2) │       │ y(3) │  输出          │
    │  └──────┘       └──────┘       └──────┘              │
    └─────────────────────────────────────────────────────────┘
    """)

    # ━━━━━ 创建 RNN 模型 ━━━━━
    input_dim = 4       # 每个词用 4 维向量表示
    hidden_dim = 8      # 记忆维度
    output_dim = 3      # 3 分类

    model = SimpleRNN(input_dim, hidden_dim, output_dim)

    print(f"  模型参数:")
    print(f"    输入维度: {input_dim}")
    print(f"    隐藏维度: {hidden_dim}")
    print(f"    输出维度: {output_dim}")

    # ━━━━━ 模拟一个序列 ━━━━━
    # 3 个时间步，每个时间步 4 维输入
    np.random.seed(42)
    seq_len = 3
    inputs = np.random.randn(seq_len, input_dim)

    print(f"\n  输入序列形状: {inputs.shape}")
    print(f"  输入序列（每个时间步 {input_dim} 维）:")
    for t in range(seq_len):
        print(f"    时间步 {t+1}: {np.array2string(inputs[t], precision=3)}")

    # 前向传播
    outputs, hiddens = model.forward(inputs)

    print(f"\n  隐藏状态（记忆）演变:")
    for t in range(seq_len):
        h_str = np.array2string(hiddens[t+1][:4], precision=3)
        print(f"    h({t+1}) = {h_str} ...")

    print(f"\n  输出（每步的预测）:")
    for t in range(seq_len):
        print(f"    y({t+1}) = {np.array2string(outputs[t], precision=3)}")

    # 用最后一个时间步的输出做预测
    final_pred = model.predict(inputs)
    print(f"\n  最终预测（取最后一步）: {np.array2string(final_pred, precision=3)}")

    # 演示反向传播
    print(f"\n  [反向传播演示]")
    targets = np.zeros((seq_len, output_dim))
    targets[-1, 0] = 1.0  # 假设正确答案是第 0 类

    gradients = model.backward(inputs, targets)
    print(f"  梯度范数（衡量学习信号的大小）:")
    for name, grad in gradients.items():
        norm = np.linalg.norm(grad)
        print(f"    {name}: {norm:.4f}")


def lesson_numpy_lstm():
    """第四部分（下）：numpy 实现的 LSTM"""

    print_separator("18.4c LSTM —— numpy 从零实现")

    print("""
    LSTM 的核心思想：有笔记本的翻译官，能选择性地记忆和遗忘

    ┌─────────────────────────────────────────────────────────┐
    │  LSTM = 遗忘门(橡皮擦) + 输入门(钢笔) + 输出门(书签)    │
    │                                                         │
    │  笔记本（细胞状态 c）: 长期记忆，信息高速公路            │
    │  工作记忆（隐藏状态 h）: 当前正在使用的信息              │
    │                                                         │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │  细胞状态 c: ──→ ──→ ──→ ──→ (信息直通)        │   │
    │  │      ↑ 擦旧写新 ↑ 擦旧写新 ↑ 擦旧写新          │   │
    │  │    遗忘+输入   遗忘+输入   遗忘+输入            │   │
    │  └─────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘
    """)

    # ━━━━━ 创建 LSTM 模型 ━━━━━
    input_dim = 4       # 每个词用 4 维向量表示
    hidden_dim = 8      # 记忆维度

    model = SimpleLSTM(input_dim, hidden_dim)

    print(f"  模型参数:")
    print(f"    输入维度: {input_dim}")
    print(f"    隐藏维度: {hidden_dim}")

    # ━━━━━ 演示单步 LSTM ━━━━━
    print(f"\n  [演示 1] 单步 LSTM:")
    x = np.random.randn(input_dim)
    h_prev = np.zeros(hidden_dim)
    c_prev = np.zeros(hidden_dim)

    h, c = model.forward(x, h_prev, c_prev)

    print(f"    输入 x: {np.array2string(x, precision=3)}")
    print(f"    输出 h: {np.array2string(h, precision=3)}")
    print(f"    细胞 c: {np.array2string(c, precision=3)}")

    # ━━━━━ 演示门的运作 ━━━━━
    print(f"\n  [演示 2] 查看各门的值:")
    concat = np.concatenate([h_prev, x])

    f_gate = model.sigmoid(np.dot(concat, model.W_f) + model.b_f)
    i_gate = model.sigmoid(np.dot(concat, model.W_i) + model.b_i)
    c_tilde = model.tanh(np.dot(concat, model.W_c) + model.b_c)
    o_gate = model.sigmoid(np.dot(concat, model.W_o) + model.b_o)

    print(f"    遗忘门 f: {np.array2string(f_gate, precision=3)}")
    print(f"    （接近 1 = 保留旧记忆，接近 0 = 擦掉旧记忆）")
    print(f"    输入门 i: {np.array2string(i_gate, precision=3)}")
    print(f"    （接近 1 = 写入新记忆，接近 0 = 不写入）")
    print(f"    候选记忆: {np.array2string(c_tilde, precision=3)}")
    print(f"    （新记忆的具体内容）")
    print(f"    输出门 o: {np.array2string(o_gate, precision=3)}")
    print(f"    （接近 1 = 输出，接近 0 = 隐藏）")

    # ━━━━━ 演示序列处理 ━━━━━
    print(f"\n  [演示 3] 处理一个序列:")
    seq_len = 4
    inputs = np.random.randn(seq_len, input_dim)

    hiddens, cells = model.forward_sequence(inputs)

    print(f"    序列长度: {seq_len}")
    print(f"    输入形状: {inputs.shape}")
    print(f"\n    隐藏状态演变（每步的记忆）:")
    for t in range(seq_len):
        h_norm = np.linalg.norm(hiddens[t])
        print(f"      h({t+1}): 范数={h_norm:.3f}, 前4维={np.array2string(hiddens[t][:4], precision=3)}")

    print(f"\n    细胞状态演变（长期记忆）:")
    for t in range(seq_len):
        c_norm = np.linalg.norm(cells[t])
        print(f"      c({t+1}): 范数={c_norm:.3f}, 前4维={np.array2string(cells[t][:4], precision=3)}")

    # ━━━━━ 对比 RNN 和 LSTM ━━━━━
    print(f"\n  [对比] RNN vs LSTM 的记忆保持能力:")
    print(f"    RNN:  h_t = tanh(W*x + W*h + b) — 每步都经过 tanh，信息衰减快")
    print(f"    LSTM: c_t = f * c_(t-1) + i * c~ — 细胞状态可以长期保持")
    print(f"    → LSTM 的细胞状态像'高速公路'，信息可以直通远方")


def lesson_pytorch():
    """第五部分：PyTorch 实战"""

    print_separator("18.5 PyTorch 实战")

    print("""
    PyTorch 是最流行的深度学习框架之一：

    ┌─────────────────────────────────────────────────────────┐
    │  核心概念：                                             │
    │  1. Tensor（张量）：多维数组，类似 numpy 的 array       │
    │  2. nn.Module：神经网络模块的基类                       │
    │  3. autograd：自动求导（自动计算梯度）                  │
    │  4. optim：优化器（SGD, Adam 等）                       │
    └─────────────────────────────────────────────────────────┘
    """)

    try_pytorch_demo()


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十八章                      ║
    ║        深度学习基础                                  ║
    ╚══════════════════════════════════════════════════════╝
    """)

    lesson_perceptron()
    lesson_activation_functions()
    lesson_forward_propagation()
    lesson_numpy_cnn()
    lesson_numpy_rnn()
    lesson_numpy_lstm()
    lesson_cnn_rnn_lstm()
    lesson_pytorch()

    # 课程总结
    print("\n" + "=" * 60)
    print("  第十八章 总结")
    print("=" * 60)
    print("""
    [OK] 感知机 — 最简单的神经网络
    [OK] 激活函数 — Sigmoid, ReLU, Tanh, Softmax
    [OK] 前向传播 — 数据从输入到输出
    [OK] CNN — numpy 从零实现（卷积、池化、前向传播）
    [OK] RNN — numpy 从零实现（前向传播、BPTT 反向传播）
    [OK] LSTM — numpy 从零实现（遗忘门、输入门、输出门）
    [OK] CNN/RNN/LSTM 概念回顾
    [OK] PyTorch — 深度学习框架实战
    """)

    print("-" * 60)
    print("  下节预告：第十九章 — 项目实战")
    print("-" * 60)
    print("""
    下一章我们将学习：
    - 聊天机器人（基于规则的对话系统）
    - 搜索引擎（完整的搜索系统）
    - 推荐系统（协同过滤 + 内容推荐）

    预习建议：
    - 回顾前面学到的所有 NLP 技术
    - 思考如何把它们组合起来解决实际问题
    """)
