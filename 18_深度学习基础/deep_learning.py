"""
==============================================================================
第十八章：深度学习基础
==============================================================================
日期：2026-05-16

同学们好！这节课我们来学习深度学习的基础知识。

----------------------------------------------------------------------
生活类比：神经网络就像"流水线工厂"
----------------------------------------------------------------------

想象你开了一家"判断水果好坏"的工厂：

  流水线有 3 个车间：
  1. 第一车间（输入层）：接收水果
  2. 第二车间（隐藏层）：检查颜色、闻气味、摸软硬
  3. 第三车间（输出层）：给出结论（好/坏）

  每个车间里有多个"工人"（神经元），
  每个工人只做一个简单的事情（比如"看颜色"），
  但合在一起就能做出复杂的判断。

神经网络就是这样：
  - 每个神经元做简单的计算
  - 大量神经元组合在一起，就能解决复杂问题

----------------------------------------------------------------------
深度学习的"深度"是什么意思？
----------------------------------------------------------------------

"深度"指的是"层数多"：
  - 浅层网络：1-2 层 → 只能学简单规律
  - 深层网络：10-100 层 → 能学复杂规律

就像人的学习：
  - 小学生：只能做简单计算
  - 大学生：能解决复杂问题
  - 层数越多，能力越强（但也越难训练）

----------------------------------------------------------------------
本章内容
----------------------------------------------------------------------

1. 感知机（最简单的神经网络）
2. 激活函数（让网络有"非线性"能力）
3. 前向传播（数据从输入到输出）
4. 反向传播（从错误中学习）
5. CNN（卷积神经网络）
6. RNN 和 LSTM
7. PyTorch 实战

==============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import random
import numpy as np


# ==============================================================================
# 第一部分：感知机（Perceptron）
# ==============================================================================
#
# 感知机是最简单的神经网络，只有 1 层。
#
# 核心思想：
#   输入 × 权重 + 偏置 → 激活函数 → 输出
#
# 生活类比：
#   想象你是一个面试官，要决定是否录用一个求职者：
#   - 学历（x1）× 权重（w1）= 学历得分
#   - 经验（x2）× 权重（w2）= 经验得分
#   - 总分 = 学历得分 + 经验得分 + 偏置
#   - 如果总分 > 阈值 → 录用
#   - 否则 → 不录用
#
# ==============================================================================


class Perceptron:
    """
    感知机 —— 最简单的神经网络

    ━━━━━━━ 生活类比 ━━━━━━━
    感知机就像一个"评委"：
    - 给每个输入打分（乘以权重）
    - 把分数加起来
    - 根据总分做出决定

    ━━━━━━━ 数学公式 ━━━━━━━
    output = activation(w1*x1 + w2*x2 + ... + wn*xn + bias)

    其中：
    - xi = 输入值
    - wi = 权重（每个输入的重要程度）
    - bias = 偏置（调整阈值）
    - activation = 激活函数
    """

    def __init__(self, n_inputs: int):
        """
        初始化感知机

        参数：
            n_inputs: 输入特征的数量
        """
        # 随机初始化权重（小随机数）
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
        # 偏置初始化为 0
        self.bias = 0

    def sigmoid(self, x: float) -> float:
        """
        Sigmoid 激活函数

        ━━━━━━━ 生活类比 ━━━━━━━
        Sigmoid 就像一个"压力表"：
        - 输入很大 → 输出接近 1（压力爆表）
        - 输入很小 → 输出接近 0（没有压力）
        - 输入中等 → 输出在 0.5 左右

        公式：σ(x) = 1 / (1 + e^(-x))
        """
        # 防止溢出
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            return math.exp(x) / (1.0 + math.exp(x))

    def forward(self, inputs: list) -> float:
        """
        前向传播：计算输出

        ━━━━━━━ 生活类比 ━━━━━━━
        就像面试官计算求职者的总分：
        1. 学历分 = 学历 × 权重1
        2. 经验分 = 经验 × 权重2
        3. 总分 = 学历分 + 经验分 + 偏置
        4. 根据总分决定是否录用

        参数：
            inputs: 输入值列表

        返回：
            输出值（0 到 1 之间）
        """
        # 计算加权和
        total = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        # 通过激活函数
        return self.sigmoid(total)

    def train(self, training_data: list, labels: list, epochs: int = 100,
              learning_rate: float = 0.1):
        """
        训练感知机

        ━━━━━━━ 生活类比 ━━━━━━━
        就像一个评委在不断调整自己的评判标准：
        1. 看一个求职者，给出判断
        2. 和正确答案对比
        3. 如果判断错了，就调整权重
        4. 重复多次，直到判断准确

        参数：
            training_data: 训练数据（每行是一个样本的特征）
            labels: 标签（每个样本的正确答案）
            epochs: 训练轮数
            learning_rate: 学习率（调整权重的步长）
        """
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_data, labels):
                # 前向传播
                output = self.forward(inputs)
                # 计算误差
                error = label - output
                total_error += error ** 2
                # 更新权重
                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * error * inputs[i]
                # 更新偏置
                self.bias += learning_rate * error

            # 每 20 轮打印一次进度
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}: 误差 = {total_error:.4f}")


# ==============================================================================
# 第二部分：激活函数
# ==============================================================================
#
# 激活函数让神经网络具有"非线性"能力。
#
# 为什么需要激活函数？
#   如果没有激活函数，无论多少层网络，都只是线性变换。
#   就像无论你叠加多少张放大镜，效果都只是"放大"，
#   但加上棱镜（非线性），就能把光分解成彩虹。
#
# ==============================================================================


def sigmoid(x: float) -> float:
    """
    Sigmoid 函数：输出范围 (0, 1)

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一个"开关"，但不是非 0 即 1，
    而是有一个平滑的过渡。
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def sigmoid_derivative(x: float) -> float:
    """
    Sigmoid 的导数

    ━━━━━━━ 生活类比 ━━━━━━━
    导数 = 变化率 = "敏感度"
    当 x 接近 0 时，导数最大（最敏感）
    当 x 很大或很小时，导数接近 0（不敏感）
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: float) -> float:
    """
    ReLU 函数：输出范围 [0, +∞)

    ━━━━━━━ 生活类比 ━━━━━━━
    ReLU 就像一个"单向阀"：
    - 正数：原样通过
    - 负数：全部变成 0

    ReLU 是目前最常用的激活函数。
    """
    return max(0, x)


def relu_derivative(x: float) -> float:
    """
    ReLU 的导数
    """
    return 1.0 if x > 0 else 0.0


def tanh(x: float) -> float:
    """
    Tanh 函数：输出范围 (-1, 1)

    ━━━━━━━ 生活类比 ━━━━━━━
    Tanh 就像一个"温度计"：
    - 正值：热（接近 1）
    - 负值：冷（接近 -1）
    - 零：常温（0）
    """
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    """
    Tanh 的导数
    """
    return 1 - math.tanh(x) ** 2


def softmax(values: list) -> list:
    """
    Softmax 函数：把一组数转换为概率分布

    ━━━━━━━ 生活类比 ━━━━━━━
    Softmax 就像"投票百分比"：
    - 原始分数：[3, 1, 2]
    - 转换后：[0.67, 0.09, 0.24]（加起来 = 1）

    常用于多分类问题的输出层。
    """
    # 防止溢出：减去最大值
    max_val = max(values)
    exp_values = [math.exp(v - max_val) for v in values]
    total = sum(exp_values)
    return [ev / total for ev in exp_values]


# ==============================================================================
# 第三部分：简单的神经网络（numpy 实现）
# ==============================================================================


class SimpleNeuralNetwork:
    """
    简单的两层神经网络

    ━━━━━━━ 生活类比 ━━━━━━━
    想象一个"流水线工厂"：
    - 输入层：原材料
    - 隐藏层：加工车间
    - 输出层：成品

    ━━━━━━━ 结构 ━━━━━━━
    输入 → [权重1] → 隐藏层 → [激活函数] → [权重2] → 输出 → [激活函数] → 最终输出
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        初始化神经网络

        参数：
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
        """
        # 随机初始化权重
        # 权重1：输入层 → 隐藏层
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
                    for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size

        # 权重2：隐藏层 → 输出层
        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)]
                    for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, inputs: list) -> tuple:
        """
        前向传播

        ━━━━━━━ 生活类比 ━━━━━━━
        数据从输入层流到输出层，就像产品在流水线上流动：
        1. 原材料进入第一个车间（输入 → 隐藏层）
        2. 第一个车间加工后，送到第二个车间（隐藏层 → 输出层）
        3. 最终得到成品（输出）

        参数：
            inputs: 输入值列表

        返回：
            (hidden_outputs, final_outputs)
        """
        # 输入层 → 隐藏层
        hidden_raw = []
        for j in range(self.hidden_size):
            total = sum(inputs[i] * self.w1[i][j] for i in range(self.input_size))
            total += self.b1[j]
            hidden_raw.append(total)

        # 隐藏层激活
        hidden_outputs = [relu(h) for h in hidden_raw]

        # 隐藏层 → 输出层
        output_raw = []
        for j in range(self.output_size):
            total = sum(hidden_outputs[i] * self.w2[i][j] for i in range(self.hidden_size))
            total += self.b2[j]
            output_raw.append(total)

        # 输出层激活（softmax 用于多分类）
        final_outputs = softmax(output_raw)

        return hidden_outputs, final_outputs

    def predict(self, inputs: list) -> list:
        """
        预测

        参数：
            inputs: 输入值列表

        返回：
            预测结果
        """
        _, outputs = self.forward(inputs)
        return outputs


# ==============================================================================
# 第四部分：CNN 概念
# ==============================================================================


def cnn_concept():
    """
    CNN（卷积神经网络）概念说明

    ━━━━━━━ 生活类比 ━━━━━━━
    CNN 就像一个"显微镜"：
    1. 卷积层：用"放大镜"扫描图像，提取局部特征
       → 就像你看一张照片，先注意到边缘、纹理
    2. 池化层：缩小图像，保留重要信息
       → 就像你把照片缩小，但还是能认出是什么
    3. 全连接层：综合所有特征，做出判断
       → 就像你看了整张照片后，说"这是一只猫"

    ━━━━━━━ 在 NLP 中的应用 ━━━━━━━
    CNN 也可以用于文本处理：
    - 卷积核在文本上滑动，提取"局部模式"
    - 比如 "不 好" 这种两字组合就是一个"特征"
    """
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    CNN 结构示意                          │
    │                                                         │
    │   输入图像 → [卷积层] → [池化层] → [全连接层] → 输出     │
    │                                                         │
    │   卷积操作：                                            │
    │   ┌───┬───┬───┐     ┌───┬───┐                          │
    │   │ 1 │ 2 │ 3 │     │ 1 │ 0 │  卷积核                  │
    │   ├───┼───┼───┤  ×  ├───┼───┤  ──→ 特征图              │
    │   │ 4 │ 5 │ 6 │     │ 0 │ 1 │                          │
    │   ├───┼───┼───┤     └───┴───┘                          │
    │   │ 7 │ 8 │ 9 │                                        │
    │   └───┴───┴───┘                                        │
    │                                                         │
    │   NLP 中的 CNN：                                        │
    │   句子："我 喜 欢 自 然 语 言 处 理"                     │
    │   卷积核大小 2：提取相邻两个词的特征                     │
    │   → "喜欢"、"自然"、"语言"、"处理"                      │
    └─────────────────────────────────────────────────────────┘
    """)


# ==============================================================================
# 第五部分：RNN 和 LSTM 概念
# ==============================================================================


def rnn_concept():
    """
    RNN（循环神经网络）概念说明

    ━━━━━━━ 生活类比 ━━━━━━━
    RNN 就像一个"有记忆力的工人"：
    - 普通工人：看一个零件，做一个判断，然后忘记
    - 有记忆力的工人：看一个零件，做一个判断，记住一些信息，
                       然后带着这些记忆去看下一个零件

    ━━━━━━━ 核心思想 ━━━━━━━
    RNN 有一个"隐藏状态"（记忆）：
    - 每一步，RNN 读取当前输入和上一步的记忆
    - 计算新的记忆
    - 输出结果
    """
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    RNN 结构示意                          │
    │                                                         │
    │   时间步 1    时间步 2    时间步 3                        │
    │   ┌─────┐    ┌─────┐    ┌─────┐                        │
    │   │  x1 │    │  x2 │    │  x3 │  输入                   │
    │   └──┬──┘    └──┬──┘    └──┬──┘                        │
    │      ↓          ↓          ↓                            │
    │   ┌─────┐    ┌─────┐    ┌─────┐                        │
    │   │  h1 │───→│  h2 │───→│  h3 │  隐藏状态（记忆）       │
    │   └──┬──┘    └──┬──┘    └──┬──┘                        │
    │      ↓          ↓          ↓                            │
    │   ┌─────┐    ┌─────┐    ┌─────┐                        │
    │   │  y1 │    │  y2 │    │  y3 │  输出                   │
    │   └─────┘    └─────┘    └─────┘                        │
    │                                                         │
    │   问题：长距离依赖 → 梯度消失/爆炸                      │
    │   解决：LSTM（长短期记忆网络）                          │
    └─────────────────────────────────────────────────────────┘
    """)


def lstm_concept():
    """
    LSTM（长短期记忆网络）概念说明

    ━━━━━━━ 生活类比 ━━━━━━━
    LSTM 就像一个"有选择性记忆的人"：
    - 遗忘门：决定忘记哪些旧信息
      → 就像你读新章节时，忘记不重要的细节
    - 输入门：决定记住哪些新信息
      → 就像你在书上划重点
    - 输出门：决定输出哪些信息
      → 就像你写读书笔记时，选择写哪些内容
    """
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    LSTM 结构示意                         │
    │                                                         │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │  细胞状态 Ct（长期记忆）                          │   │
    │   │  ────────→ ────────→ ────────→                   │   │
    │   └─────────────────────────────────────────────────┘   │
    │           ↑          ↑          ↑                        │
    │       遗忘门      输入门      输出门                      │
    │                                                         │
    │   遗忘门：f = σ(Wf · [h(t-1), x(t)] + bf)              │
    │   输入门：i = σ(Wi · [h(t-1), x(t)] + bi)              │
    │   输出门：o = σ(Wo · [h(t-1), x(t)] + bo)              │
    │                                                         │
    │   细胞状态：Ct = f * C(t-1) + i * tanh(...)             │
    │   隐藏状态：ht = o * tanh(Ct)                           │
    └─────────────────────────────────────────────────────────┘
    """)


# ==============================================================================
# 第六部分：numpy 实现 CNN / RNN / LSTM
# ==============================================================================
#
# 接下来我们用 numpy 从零实现三种经典的深度学习模型。
# 为什么要用 numpy 而不是直接用 PyTorch？
#   因为只有亲手实现，你才能真正理解底层原理！
#   就像学做菜，先要学会手工切菜，才能用好料理机。
#
# ==============================================================================


class SimpleCNN:
    """
    简单的 1D 卷积神经网络（用于文本分类）

    ━━━━━━━ 生活类比 ━━━━━━━
    CNN 就像一个拿着"放大镜"的侦探：
    - 他不会一开始就看整篇文章
    - 而是拿着放大镜，从左到右，一小段一小段地扫描
    - 每次只关注几个相邻的词，寻找"线索"（模式）
    - 最后把所有线索汇总，得出结论

    ━━━━━━━ 核心思想 ━━━━━━━
    1. 词嵌入（Embedding）：把单词变成数字向量
       → 就像给每个词发一张"身份证"（一组数字）
    2. 卷积（Convolution）：用小窗口滑动提取局部特征
       → 就像用放大镜逐段扫描，看"不 好"这种组合
    3. 池化（Pooling）：取每段中最强的信号
       → 就像从一堆线索中，只保留最关键的那条
    4. 全连接（Dense）：综合所有信号做最终判断
       → 就像侦探汇总所有线索，得出"这是正面评价"的结论

    ━━━━━━━ 在 NLP 中的应用 ━━━━━━━
    文本 CNN 特别擅长捕捉"局部模式"：
    - "不 好" → 负面模式
    - "非常 棒" → 正面模式
    - "虽然...但是" → 转折模式
    """

    def __init__(self, vocab_size, embed_dim, num_filters, filter_size):
        """
        初始化 CNN

        参数：
            vocab_size:  词汇表大小（有多少个不同的词）
            embed_dim:   词嵌入维度（每个词用几个数字表示）
            num_filters: 卷积核数量（提取多少种不同的特征）
            filter_size: 卷积核大小（每次看几个词）

        生活类比：
            vocab_size = 字典里有多少个字
            embed_dim  = 每个字用几个属性来描述（笔画数、拼音长度...）
            num_filters = 有几个侦探，每人关注不同的线索类型
            filter_size = 放大镜的窗口大小（每次看几个字）
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_size = filter_size

        # ━━━━━━━ 词嵌入矩阵 ━━━━━━━
        # 就像一本"字典"，每个词对应一个向量
        # 初始化时用随机小数，训练过程中会不断调整
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1

        # ━━━━━━━ 卷积核（滤波器）━━━━━━━
        # 就像侦探的"放大镜"，每个卷积核关注一种模式
        # 形状：(num_filters, filter_size, embed_dim)
        # 含义：每个卷积核覆盖 filter_size 个词，每个词有 embed_dim 维
        self.filters = np.random.randn(num_filters, filter_size, embed_dim) * 0.1

        # ━━━━━━━ 卷积层偏置 ━━━━━━━
        # 每个卷积核有一个偏置，就像侦探的"个人偏好"
        self.conv_bias = np.zeros(num_filters)

        # ━━━━━━━ 全连接层权重 ━━━━━━━
        # 池化后的特征 → 分类输出
        # 就像侦探汇总线索后，做出最终判断的"决策权重"
        # 这里简化为二分类（正面/负面），输出 2 个值
        self.fc_weights = np.random.randn(num_filters, 2) * 0.1
        self.fc_bias = np.zeros(2)

    def conv1d(self, input_matrix, filters):
        """
        一维卷积操作 —— CNN 的核心！

        ━━━━━━━ 生活类比 ━━━━━━━
        想象你在读一句话："我 非常 喜欢 这部 电影"
        你拿着一个"窗口大小=2"的放大镜：
        - 先看 "我 非常"    → 计算一个分数
        - 再看 "非常 喜欢"  → 计算一个分数
        - 再看 "喜欢 这部"  → 计算一个分数
        - 再看 "这部 电影"  → 计算一个分数

        每次计算就是：把窗口内的词向量和卷积核做"点积"
        点积越大，说明这段文本和卷积核越"匹配"

        ━━━━━━━ 数学公式 ━━━━━━━
        对于输入序列的每个位置 t：
            output[t] = Σ (input[t:t+filter_size] * filter) + bias

        参数：
            input_matrix: 输入矩阵，形状 (seq_len, embed_dim)
                          就像一句话中每个词的向量排成一排
            filters:      卷积核，形状 (num_filters, filter_size, embed_dim)
                          就像多个不同的放大镜

        返回：
            feature_maps: 特征图，形状 (num_filters, output_len)
                          每个卷积核在每个位置的得分
        """
        seq_len = input_matrix.shape[0]      # 句子长度（有多少个词）
        num_filters = filters.shape[0]       # 卷积核数量
        filter_size = filters.shape[1]       # 卷积核窗口大小
        embed_dim = filters.shape[2]         # 词嵌入维度

        # 计算输出长度：窗口滑动后能停多少个位置
        # 比如句子长 5，窗口大小 2，那就能停 4 个位置
        output_len = seq_len - filter_size + 1

        # 初始化特征图：每个卷积核在每个位置产生一个分数
        feature_maps = np.zeros((num_filters, output_len))

        # ━━━━━━━ 开始卷积！━━━━━━━
        # 对每个卷积核（每个侦探）
        for f_idx in range(num_filters):
            # 对每个滑动位置（放大镜停在每个位置）
            for t in range(output_len):
                # 取出当前位置的窗口内容
                # 就像放大镜框住了这几个词
                window = input_matrix[t:t + filter_size, :]

                # 计算点积：窗口内容和卷积核的"匹配程度"
                # 点积 = 对应元素相乘再求和
                # 匹配度越高，值越大
                dot_product = np.sum(window * filters[f_idx])

                # 加上偏置，得到这个位置的特征值
                feature_maps[f_idx, t] = dot_product + self.conv_bias[f_idx]

        return feature_maps

    def max_pool(self, feature_map):
        """
        最大池化 —— 提取最强信号

        ━━━━━━━ 生活类比 ━━━━━━━
        想象你有一排温度计读数：[23, 35, 28, 31]
        最大池化就是只保留最高温度：35

        为什么要这样做？
        - 卷积后我们得到了每个位置的特征值
        - 但我们不需要每个位置的精确信息
        - 我们只想知道"最强的特征出现在哪里"
        - 就像你问"这句话里最强烈的感情是什么"，
          而不是"每个位置的感情分别是什么"

        ━━━━━━━ 数学公式 ━━━━━━━
        pooled = max(feature_map[0], feature_map[1], ..., feature_map[n])

        参数：
            feature_map: 特征图，形状 (num_filters, output_len)

        返回：
            pooled: 池化结果，形状 (num_filters,)
                    每个卷积核只保留一个最大值
        """
        # axis=1 表示沿着"时间维度"取最大值
        # 就像在每个侦探的记录中，只保留最亮的那条线索
        return np.max(feature_map, axis=1)

    def forward(self, word_indices):
        """
        前向传播 —— 从单词到分类结果

        ━━━━━━━ 完整流程 ━━━━━━━
        就像侦探破案的完整流程：

        1. 查字典（Embedding）：
           把 "不好" 变成 [[0.2, -0.5], [0.8, 0.1]]

        2. 扫描线索（Conv1d）：
           用放大镜逐段扫描，得到每个位置的"匹配分数"

        3. 激活（ReLU）：
           去掉负分（负面线索不算数），只保留正分

        4. 汇总（MaxPool）：
           每种线索只保留最强的那条

        5. 判决（Dense + Softmax）：
           综合所有线索，计算"正面"和"负面"的概率

        参数：
            word_indices: 单词索引列表，如 [3, 7, 2, 15]
                          表示句子由第 3、7、2、15 号词组成

        返回：
            probs: 分类概率，如 [0.8, 0.2] 表示 80% 正面
        """
        # ━━━━ 第一步：词嵌入 ━━━━
        # 把词的编号变成词向量
        # 就像查字典，把"字"变成"释义"
        embedded = self.embedding[word_indices]
        # embedded 的形状：(seq_len, embed_dim)

        # ━━━━ 第二步：卷积 ━━━━
        # 用放大镜扫描，提取局部特征
        feature_maps = self.conv1d(embedded, self.filters)
        # feature_maps 的形状：(num_filters, output_len)

        # ━━━━ 第三步：ReLU 激活 ━━━━
        # 去掉负值，只保留"阳性线索"
        # 就像侦探只关注"支持某个结论"的证据
        feature_maps = np.maximum(0, feature_maps)

        # ━━━━ 第四步：最大池化 ━━━━
        # 每种线索只保留最强信号
        pooled = self.max_pool(feature_maps)
        # pooled 的形状：(num_filters,)

        # ━━━━ 第五步：全连接层 ━━━━
        # 综合所有线索，计算最终得分
        # 就像侦探把所有线索放在一起，做出最终判断
        logits = np.dot(pooled, self.fc_weights) + self.fc_bias
        # logits 的形状：(2,) —— 两个类别的原始得分

        # ━━━━ 第六步：Softmax ━━━━
        # 把得分转换为概率
        # 就像把"得分"转换为"可能性百分比"
        # 减去最大值防止溢出（这是数值计算的小技巧）
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return probs


class SimpleRNN:
    """
    简单的循环神经网络（RNN）—— 从零实现

    ━━━━━━━ 生活类比 ━━━━━━━
    RNN 就像一个"有记性的翻译官"：
    - 他逐词翻译一句话
    - 翻译每个词时，他不仅看当前的词，还"记得"之前翻译过的内容
    - 比如翻译 "我 喜欢 苹果" 时：
      - 看到 "我" → 翻译 "I"，记住"主语是第一人称"
      - 看到 "喜欢" → 翻译 "like"，记住"动词是喜欢"
      - 看到 "苹果" → 翻译 "apples"，结合记忆理解这是宾语

    ━━━━━━━ 核心思想 ━━━━━━━
    RNN 有一个"隐藏状态"（hidden state），就是它的"记忆"：
    - 每一步：新的记忆 = f(当前输入, 旧的记忆)
    - 数学公式：h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)

    ━━━━━━━ 问题：梯度消失 ━━━━━━━
    RNN 的记忆会随时间"衰减"：
    - 就像传话游戏，传了 100 个人后，原话已经面目全非
    - 距离越远的信息，梯度越小，越难学到
    - 这就是为什么需要 LSTM（后面会讲）
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化 RNN

        参数：
            input_dim:  输入维度（每个时间步的输入大小）
            hidden_dim: 隐藏层维度（记忆的大小）
            output_dim: 输出维度（分类类别数）

        生活类比：
            input_dim  = 翻译官每步看到的"信息量"
            hidden_dim = 翻译官的"记忆容量"
            output_dim = 最终输出的"选项数"
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ━━━━━━━ 权重初始化 ━━━━━━━
        # 使用 Xavier 初始化，让梯度传播更稳定
        # 就像给翻译官一个"合理的起点"

        # W_xh：输入 → 隐藏层的权重
        # "当前看到的信息如何影响记忆"
        scale_xh = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale_xh

        # W_hh：隐藏层 → 隐藏层的权重（循环连接）
        # "旧的记忆如何影响新的记忆"
        # 这是 RNN 的关键！它让信息能在时间步之间传递
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale_hh

        # W_hy：隐藏层 → 输出层的权重
        # "记忆如何转换为输出"
        scale_hy = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.W_hy = np.random.randn(hidden_dim, output_dim) * scale_hy

        # 偏置
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, inputs):
        """
        前向传播 —— 逐时间步处理序列

        ━━━━━━━ 生活类比 ━━━━━━━
        就像翻译官逐词翻译：
        1. 读第 1 个词，更新记忆，给出输出
        2. 读第 2 个词，结合旧记忆更新记忆，给出输出
        3. 读第 3 个词，结合旧记忆更新记忆，给出输出
        4. ... 直到读完所有词

        ━━━━━━━ 数学公式 ━━━━━━━
        对于每个时间步 t：
            h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            y_t = W_hy * h_t + b_y

        参数：
            inputs: 输入序列，形状 (seq_len, input_dim)
                    就像一句话中每个词的向量

        返回：
            outputs:   每个时间步的输出，形状 (seq_len, output_dim)
            hiddens:   每个时间步的隐藏状态，形状 (seq_len, hidden_dim)
                       这些是"记忆"，用于反向传播
        """
        seq_len = inputs.shape[0]

        # 存储每个时间步的隐藏状态（用于反向传播）
        hiddens = np.zeros((seq_len + 1, self.hidden_dim))
        # 存储每个时间步的输出
        outputs = np.zeros((seq_len, self.output_dim))

        # 初始隐藏状态为零（翻译官刚开始没有记忆）
        h_prev = np.zeros(self.hidden_dim)

        # ━━━━━━ 逐时间步处理 ━━━━━━
        for t in range(seq_len):
            # 当前输入
            x_t = inputs[t]

            # ━━━━━ 计算新的隐藏状态 ━━━━━
            # 公式：h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            #
            # 翻译：
            # - W_xh * x_t      → 当前输入带来的"新信息"
            # - W_hh * h_prev    → 旧记忆带来的"历史信息"
            # - b_h              → 翻译官的"个人偏好"
            # - tanh(...)        → 把结果压缩到 (-1, 1) 之间
            #
            # 就像翻译官综合"当前看到的词"和"之前翻译的内容"，
            # 得出新的理解
            h_raw = np.dot(x_t, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
            h_t = np.tanh(h_raw)

            # ━━━━━ 计算输出 ━━━━━
            # 公式：y_t = W_hy * h_t + b_y
            # 把当前记忆转换为输出
            y_t = np.dot(h_t, self.W_hy) + self.b_y

            # 保存状态
            hiddens[t] = h_prev  # 保存旧的隐藏状态
            outputs[t] = y_t
            h_prev = h_t         # 更新记忆

        # 保存最后一个隐藏状态
        hiddens[seq_len] = h_prev

        return outputs, hiddens

    def backward(self, inputs, targets):
        """
        反向传播（BPTT）—— 从错误中学习

        ━━━━━━━ 生活类比 ━━━━━━━
        就像翻译官翻译完一整句话后，回头看自己的翻译：
        - 对比正确答案，发现哪里翻译错了
        - 从最后一个词开始，逐词往前回溯
        - 每个词都计算"我的错误对这个词的翻译有多大影响"
        - 然后调整翻译策略

        ━━━━━━━ 核心思想 ━━━━━━━
        BPTT（Backpropagation Through Time）：
        - 把 RNN "展开"成一个很深的前馈网络
        - 从最后一个时间步开始，逐个往前计算梯度
        - 所有时间步共享同一组权重

        ━━━━━━━ 梯度消失问题 ━━━━━━━
        - 当序列很长时，梯度要经过很多次乘法
        - 如果每次乘法都让梯度变小一点，最终梯度会趋近于 0
        - 这就是"梯度消失"——远距离的信息学不到
        - 解决方案：梯度裁剪（gradient clipping）

        参数：
            inputs:  输入序列，形状 (seq_len, input_dim)
            targets: 目标输出，形状 (seq_len, output_dim)

        返回：
            gradients: 包含所有权重梯度的字典
        """
        seq_len = inputs.shape[0]

        # 先做前向传播，获取所有中间状态
        outputs, hiddens = self.forward(inputs)

        # 初始化梯度
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # 下一个时间步传回来的梯度（初始为零）
        dh_next = np.zeros(self.hidden_dim)

        # ━━━━━━ 从后往前遍历 ━━━━━━
        # 就像翻译官从句子末尾开始回溯检查
        for t in range(seq_len - 1, -1, -1):
            # ━━━━━ 输出层的梯度 ━━━━━
            # 计算预测值和真实值的差距
            dy = outputs[t] - targets[t]

            # ━━━━━ 隐藏层 → 输出层的梯度 ━━━━━
            # dW_hy = dy * h_t^T
            dW_hy += np.outer(hiddens[t + 1], dy)
            db_y += dy

            # ━━━━━ 隐藏层的梯度 ━━━━━
            # 来自两部分：输出层的梯度 + 下一个时间步传回来的梯度
            dh = np.dot(self.W_hy, dy) + dh_next

            # ━━━━━ tanh 的导数 ━━━━━
            # tanh'(x) = 1 - tanh(x)^2
            # hiddens[t+1] 就是 tanh 的输出
            dh_raw = dh * (1 - hiddens[t + 1] ** 2)

            # ━━━━━ 累加权重梯度 ━━━━━
            # W_xh 的梯度：输入 × dh_raw
            dW_xh += np.outer(inputs[t], dh_raw)
            # W_hh 的梯度：上一步隐藏状态 × dh_raw
            dW_hh += np.outer(hiddens[t], dh_raw)
            # 偏置梯度
            db_h += dh_raw

            # ━━━━━ 传递给上一个时间步 ━━━━━
            # 这就是"记忆"的反向传播
            dh_next = np.dot(self.W_hh, dh_raw)

        # ━━━━━━ 梯度裁剪 ━━━━━━
        # 防止梯度爆炸：把梯度限制在 [-5, 5] 范围内
        # 就像给翻译官的"调整幅度"设一个上限，
        # 防止他因为一次大错就彻底改变翻译风格
        for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(grad, -5, 5, out=grad)

        gradients = {
            'dW_xh': dW_xh,
            'dW_hh': dW_hh,
            'dW_hy': dW_hy,
            'db_h': db_h,
            'db_y': db_y
        }

        return gradients

    def predict(self, inputs):
        """
        预测：处理输入序列，返回最后一步的输出

        参数：
            inputs: 输入序列，形状 (seq_len, input_dim)

        返回：
            最后一步的输出，形状 (output_dim,)
        """
        outputs, _ = self.forward(inputs)
        return outputs[-1]  # 返回最后一个时间步的输出


class SimpleLSTM:
    """
    长短期记忆网络（LSTM）—— 从零实现

    ━━━━━━━ 生活类比 ━━━━━━━
    LSTM 就像一个"有笔记本的翻译官"：

    普通 RNN 翻译官：
    - 只靠脑子记，记忆力有限
    - 长句子翻译到后面，前面的内容就忘了

    LSTM 翻译官：
    - 手里有一个笔记本（细胞状态）
    - 有三件工具：
      1. 橡皮擦（遗忘门）：擦掉不需要的旧笔记
      2. 钢笔（输入门）：写下重要的新信息
      3. 书签（输出门）：标记当前需要参考的笔记

    ━━━━━━━ 核心思想 ━━━━━━━
    LSTM 有两个"状态"：
    - 隐藏状态 h：短期记忆，当前的"工作记忆"
    - 细胞状态 c：长期记忆，笔记本上的"持久记录"

    三个门控制信息流动：
    1. 遗忘门：决定擦掉哪些旧记忆
    2. 输入门：决定写入哪些新记忆
    3. 输出门：决定输出哪些记忆

    ━━━━━━━ 为什么 LSTM 能解决长期依赖？ ━━━━━━━
    普通 RNN 的记忆每一步都会被"完全覆盖"：
    - h_t = tanh(...) → 信息每步都在变换
    - 长距离信息经过多次变换后就消失了

    LSTM 的细胞状态可以"直通"：
    - c_t = f * c_{t-1} + i * c_tilde
    - 如果遗忘门 f ≈ 1，旧信息几乎原封不动地传下去
    - 这就像笔记本上的内容，不擦就一直在
    """

    def __init__(self, input_dim, hidden_dim):
        """
        初始化 LSTM

        参数：
            input_dim:  输入维度
            hidden_dim: 隐藏层维度（记忆容量）

        生活类比：
            input_dim  = 每步看到的"信息量"
            hidden_dim = 笔记本的"页数"（记忆容量）
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # ━━━━━━━ 权重初始化 ━━━━━━━
        # LSTM 有 4 组权重，分别对应 3 个门 + 1 个候选记忆
        # 每组权重包含：输入→门 和 隐藏层→门 两部分
        # 我们把它们合并成一个大矩阵，方便计算

        # 合并输入维度和隐藏维度
        concat_dim = input_dim + hidden_dim

        # Xavier 初始化的标准差
        scale = np.sqrt(2.0 / (concat_dim + hidden_dim))

        # ━━━━━ 遗忘门（Forget Gate）━━━━━
        # 决定"擦掉哪些旧笔记"
        # f = sigmoid(W_f * [h_prev, x] + b_f)
        self.W_f = np.random.randn(concat_dim, hidden_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        # 初始化偏置为 1：默认"不遗忘"（让训练初期信息流通顺畅）
        self.b_f[:] = 1.0

        # ━━━━━ 输入门（Input Gate）━━━━━
        # 决定"写下哪些新信息"
        # i = sigmoid(W_i * [h_prev, x] + b_i)
        self.W_i = np.random.randn(concat_dim, hidden_dim) * scale
        self.b_i = np.zeros(hidden_dim)

        # ━━━━━ 候选记忆（Candidate）━━━━━
        # "新记忆的内容"
        # c_tilde = tanh(W_c * [h_prev, x] + b_c)
        self.W_c = np.random.randn(concat_dim, hidden_dim) * scale
        self.b_c = np.zeros(hidden_dim)

        # ━━━━━ 输出门（Output Gate）━━━━━
        # 决定"输出哪些记忆"
        # o = sigmoid(W_o * [h_prev, x] + b_o)
        self.W_o = np.random.randn(concat_dim, hidden_dim) * scale
        self.b_o = np.zeros(hidden_dim)

    def sigmoid(self, x):
        """
        Sigmoid 激活函数

        ━━━━━━━ 生活类比 ━━━━━━━
        就像一个"阀门"：
        - 输出接近 1 → 阀门打开，信息通过
        - 输出接近 0 → 阀门关闭，信息被阻断
        - 输出 0.5   → 半开半闭

        公式：σ(x) = 1 / (1 + e^(-x))
        """
        # 使用 np.clip 防止溢出
        # 就像给阀门加了安全限制，防止压力过大
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        """
        Tanh 激活函数

        ━━━━━━━ 生活类比 ━━━━━━━
        就像一个"温度计"：
        - 输出接近 1  → 很热（强正信号）
        - 输出接近 -1 → 很冷（强负信号）
        - 输出 0      → 常温（中性）

        公式：tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        """
        return np.tanh(x)

    def forward(self, x, prev_h, prev_c):
        """
        单步前向传播 —— LSTM 的核心！

        ━━━━━━━ 完整流程 ━━━━━━━
        想象翻译官在一个时间步的工作：

        1. 遗忘门（橡皮擦）：
           翻译官看了看笔记本，决定擦掉哪些旧内容
           → "之前记的那个人名跟这句话无关，擦掉"

        2. 输入门（钢笔）：
           翻译官决定写下哪些新信息
           → "这句话的主语很重要，记下来"

        3. 候选记忆（新笔记的内容）：
           翻译官把新信息写在草稿纸上
           → "主语是'小明'"

        4. 更新笔记本（细胞状态）：
           旧笔记 × 遗忘比例 + 新笔记 × 记录比例
           → "擦掉不重要的，加上新发现的"

        5. 输出门（书签）：
           翻译官决定当前需要参考笔记的哪部分
           → "现在需要参考主语信息，其他先不管"

        6. 最终输出（隐藏状态）：
           当前时间步的输出 = 输出门 × 处理后的笔记本

        ━━━━━━━ 数学公式 ━━━━━━━
        concat = [h_prev, x]                    # 拼接上一步隐藏状态和当前输入
        f = sigmoid(W_f * concat + b_f)          # 遗忘门
        i = sigmoid(W_i * concat + b_i)          # 输入门
        c_tilde = tanh(W_c * concat + b_c)       # 候选记忆
        c = f * prev_c + i * c_tilde             # 更新细胞状态
        o = sigmoid(W_o * concat + b_o)          # 输出门
        h = o * tanh(c)                          # 最终隐藏状态

        参数：
            x:       当前输入，形状 (input_dim,)
            prev_h:  上一步隐藏状态，形状 (hidden_dim,)
            prev_c:  上一步细胞状态，形状 (hidden_dim,)

        返回：
            h:       当前隐藏状态，形状 (hidden_dim,)
            c:       当前细胞状态，形状 (hidden_dim,)
        """
        # ━━━━━ 第一步：拼接输入 ━━━━━
        # 把上一步的记忆和当前输入合并
        # 就像翻译官同时看"笔记本"和"当前的词"
        concat = np.concatenate([prev_h, x])

        # ━━━━━ 第二步：遗忘门 ━━━━━
        # "橡皮擦" —— 决定擦掉哪些旧记忆
        # sigmoid 输出 0~1：0 = 完全擦掉，1 = 完全保留
        # 就像翻译官翻看旧笔记，决定哪些内容不再需要
        f = self.sigmoid(np.dot(concat, self.W_f) + self.b_f)

        # ━━━━━ 第三步：输入门 ━━━━━
        # "钢笔" —— 决定写下哪些新信息
        # sigmoid 输出 0~1：0 = 不写，1 = 写入
        # 就像翻译官决定"这个信息值得记下来"
        i = self.sigmoid(np.dot(concat, self.W_i) + self.b_i)

        # ━━━━━ 第四步：候选记忆 ━━━━━
        # "新笔记的内容" —— 可能写入笔记本的信息
        # tanh 输出 -1~1：新记忆的具体内容
        # 就像翻译官在草稿纸上写下新发现
        c_tilde = self.tanh(np.dot(concat, self.W_c) + self.b_c)

        # ━━━━━ 第五步：更新细胞状态 ━━━━━
        # "更新笔记本" —— LSTM 最核心的操作！
        #
        # 公式：c = f * prev_c + i * c_tilde
        #
        # 翻译：
        # - f * prev_c     → 旧笔记中保留下来的部分（没被擦掉的）
        # - i * c_tilde    → 新写入的笔记
        # - 两者相加       → 更新后的完整笔记
        #
        # 这就是 LSTM 能记住长期信息的秘密！
        # 如果 f ≈ 1 且 i ≈ 0：旧笔记几乎原封不动传下去
        # 如果 f ≈ 0 且 i ≈ 1：完全用新笔记替代旧笔记
        c = f * prev_c + i * c_tilde

        # ━━━━━ 第六步：输出门 ━━━━━
        # "书签" —— 决定输出笔记的哪部分
        # sigmoid 输出 0~1：0 = 不输出，1 = 输出
        # 就像翻译官决定"现在需要用笔记中的哪部分"
        o = self.sigmoid(np.dot(concat, self.W_o) + self.b_o)

        # ━━━━━ 第七步：计算隐藏状态 ━━━━━
        # "最终输出" —— 当前时间步的输出
        #
        # 公式：h = o * tanh(c)
        #
        # 翻译：
        # - tanh(c) → 把细胞状态压缩到 (-1, 1)
        # - o * ...  → 只输出被"书签"标记的部分
        #
        # 就像翻译官把笔记本中需要的部分摘出来，
        # 作为当前翻译的参考
        h = o * self.tanh(c)

        return h, c

    def forward_sequence(self, inputs):
        """
        处理整个序列 —— 逐时间步调用 forward

        ━━━━━━━ 生活类比 ━━━━━━━
        就像翻译官逐词翻译一整句话：
        - 每翻译一个词，就更新一次笔记本
        - 翻译完所有词后，返回整个过程中产生的所有输出

        参数：
            inputs: 输入序列，形状 (seq_len, input_dim)

        返回：
            hiddens: 所有时间步的隐藏状态，形状 (seq_len, hidden_dim)
            cells:   所有时间步的细胞状态，形状 (seq_len, hidden_dim)
        """
        seq_len = inputs.shape[0]

        # 初始化：翻译官刚开始，笔记本是空白的
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        # 存储每个时间步的状态
        hiddens = np.zeros((seq_len, self.hidden_dim))
        cells = np.zeros((seq_len, self.hidden_dim))

        # ━━━━━━ 逐时间步处理 ━━━━━━
        for t in range(seq_len):
            # 处理当前词，更新记忆
            h, c = self.forward(inputs[t], h, c)
            # 保存状态
            hiddens[t] = h
            cells[t] = c

        return hiddens, cells


# ==============================================================================
# 第七部分：PyTorch 实战
# ==============================================================================


def try_pytorch_demo():
    """
    尝试使用 PyTorch 构建简单的神经网络

    ━━━━━━━ 注意 ━━━━━━━
    这个函数会尝试导入 PyTorch。
    如果没有安装 PyTorch，会打印概念说明。
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # ━━━━━━━ 示例 1：简单的全连接网络 ━━━━━━━
        print("  [示例 1] 简单的全连接网络:")

        class SimpleNet(nn.Module):
            """简单的两层网络"""

            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)   # 输入 4 维，隐藏层 8 维
                self.fc2 = nn.Linear(8, 3)   # 隐藏层 8 维，输出 3 维
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleNet()
        print(f"    模型结构:\n{model}")

        # 测试前向传播
        x = torch.randn(1, 4)  # 1 个样本，4 维特征
        output = model(x)
        print(f"    输入形状: {x.shape}")
        print(f"    输出形状: {output.shape}")

        # ━━━━━━━ 示例 2：文本分类 CNN ━━━━━━━
        print("\n  [示例 2] 文本分类 CNN:")

        class TextCNN(nn.Module):
            """
            用于文本分类的 CNN

            ━━━━━━━ 原理 ━━━━━━━
            1. 词嵌入：把词转换为向量
            2. 卷积：提取局部特征（相邻词的组合）
            3. 池化：保留最重要的特征
            4. 全连接：输出分类结果
            """

            def __init__(self, vocab_size: int, embed_dim: int = 32,
                         num_classes: int = 3):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                # 不同大小的卷积核，捕捉不同长度的 n-gram
                self.conv1 = nn.Conv1d(embed_dim, 16, kernel_size=2, padding=1)
                self.conv2 = nn.Conv1d(embed_dim, 16, kernel_size=3, padding=1)
                self.fc = nn.Linear(32, num_classes)
                self.relu = nn.ReLU()

            def forward(self, x):
                embedded = self.embedding(x)  # (batch, seq, embed)
                embedded = embedded.permute(0, 2, 1)  # (batch, embed, seq)

                # 卷积
                c1 = self.relu(self.conv1(embedded))  # (batch, 16, seq)
                c2 = self.relu(self.conv2(embedded))  # (batch, 16, seq)

                # 池化（取最大值）
                p1 = torch.max(c1, dim=2)[0]  # (batch, 16)
                p2 = torch.max(c2, dim=2)[0]  # (batch, 16)

                # 拼接
                cat = torch.cat([p1, p2], dim=1)  # (batch, 32)

                # 全连接
                output = self.fc(cat)
                return output

        vocab_size = 100
        model_cnn = TextCNN(vocab_size)
        x = torch.randint(0, vocab_size, (2, 10))  # 2 个样本，每个 10 个词
        output = model_cnn(x)
        print(f"    CNN 输入形状: {x.shape}")
        print(f"    CNN 输出形状: {output.shape}")

        # ━━━━━━━ 示例 3：简单 RNN ━━━━━━━
        print("\n  [示例 3] 简单 RNN:")

        class SimpleRNN(nn.Module):
            """简单的 RNN 用于序列处理"""

            def __init__(self, vocab_size: int, embed_dim: int = 32,
                         hidden_dim: int = 32, num_classes: int = 3):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                embedded = self.embedding(x)
                output, hidden = self.rnn(embedded)
                # 取最后一个时间步的输出
                last_output = output[:, -1, :]
                return self.fc(last_output)

        model_rnn = SimpleRNN(vocab_size)
        x = torch.randint(0, vocab_size, (2, 10))
        output = model_rnn(x)
        print(f"    RNN 输入形状: {x.shape}")
        print(f"    RNN 输出形状: {output.shape}")

        return True

    except ImportError:
        print("  [提示] PyTorch 未安装，跳过代码演示")
        print("  安装方式: pip install torch")
        return False


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

    # 感知机演示
    print("=" * 60)
    print("  1. 感知机演示（学习 AND 逻辑）")
    print("=" * 60)

    # AND 逻辑的训练数据
    and_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    and_labels = [0, 0, 0, 1]

    p = Perceptron(2)
    print("  训练前:")
    for inputs in and_data:
        print(f"    {inputs} → {p.forward(inputs):.4f}")

    print("\n  训练中:")
    p.train(and_data, and_labels, epochs=100, learning_rate=0.5)

    print("\n  训练后:")
    for inputs in and_data:
        output = p.forward(inputs)
        print(f"    {inputs} → {output:.4f} (期望: {and_labels[and_data.index(inputs)]})")

    # 激活函数演示
    print("\n" + "=" * 60)
    print("  2. 激活函数演示")
    print("=" * 60)

    test_values = [-2, -1, 0, 1, 2]
    print(f"  {'输入':<8} {'Sigmoid':<12} {'ReLU':<8} {'Tanh':<8}")
    print(f"  {'-'*36}")
    for x in test_values:
        print(f"  {x:<8} {sigmoid(x):<12.4f} {relu(x):<8.1f} {tanh(x):<8.4f}")

    # Softmax 演示
    print(f"\n  Softmax 示例:")
    values = [2.0, 1.0, 0.5]
    probs = softmax(values)
    print(f"    输入: {values}")
    print(f"    输出: {[f'{p:.3f}' for p in probs]}")
    print(f"    总和: {sum(probs):.3f}")

    # 神经网络演示
    print("\n" + "=" * 60)
    print("  3. 简单神经网络演示")
    print("=" * 60)

    nn_model = SimpleNeuralNetwork(3, 4, 2)
    inputs = [1.0, 0.5, -0.5]
    hidden, output = nn_model.forward(inputs)
    print(f"  输入: {inputs}")
    print(f"  隐藏层输出: {[f'{h:.3f}' for h in hidden]}")
    print(f"  最终输出: {[f'{o:.3f}' for o in output]}")

    # CNN 概念
    print("\n" + "=" * 60)
    print("  4. CNN 概念")
    print("=" * 60)
    cnn_concept()

    # RNN/LSTM 概念
    print("=" * 60)
    print("  5. RNN 和 LSTM 概念")
    print("=" * 60)
    rnn_concept()
    lstm_concept()

    # PyTorch 演示
    print("=" * 60)
    print("  6. PyTorch 实战演示")
    print("=" * 60)
    try_pytorch_demo()

    # =============================================
    # 课程总结
    # =============================================
    """
    核心收获：
    - 感知机是最简单的神经网络 —— 加权求和加激活函数，理解了它就理解了深度学习的起点
    - 激活函数赋予网络非线性能力 —— 没有激活函数，多层网络等价于单层线性变换
    - LSTM 的三个门（遗忘门、输入门、输出门）协同工作 —— 让网络能选择性地记忆和遗忘

    常见陷阱：
    - 梯度消失/爆炸 —— 深层网络中梯度连乘导致信号衰减或膨胀，需要用梯度裁剪和合适的初始化
    - 激活函数选择不当 —— Sigmoid 在深层网络中容易梯度消失，ReLU 是目前最常用的默认选择
    - 忽略数值稳定性 —— Softmax 和 Sigmoid 计算时要注意减去最大值防止溢出
    """
