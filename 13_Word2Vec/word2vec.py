"""
==============================================================================
第十三章：Word2Vec 词向量
==============================================================================
日期：2026-05-16

同学们好！前面我们学了 TF-IDF，它能把文档表示为向量。
但是 TF-IDF 有一个很大的缺点：它无法捕捉词与词之间的语义关系。
比如"国王"和"女王"意思很接近，但 TF-IDF 完全看不出来。

今天我们学习 Word2Vec —— 给每个词一个"身份证"！

----------------------------------------------------------------------
生活类比：给每个词一张身份证
----------------------------------------------------------------------

想象一个城市里有 10 万居民，每个人都要办身份证。
身份证上有关键信息：性别、年龄、职业、学历、收入...

  张三的身份证: [男, 25, 工程师, 本科, 15000]
  李四的身份证: [男, 28, 工程师, 硕士, 18000]
  王五的身份证: [女, 30, 医生,   博士, 20000]

通过身份证信息，我们可以判断：
  - 张三和李四很像（都是年轻男工程师）
  - 王五和他们不太一样（是女医生）

Word2Vec 也是一样的道理：给每个词一张"数字身份证"（词向量）：

  "国王" → [0.2, 0.8, 0.1, -0.3, 0.5]
  "女王" → [0.3, 0.7, 0.1, -0.2, 0.6]   ← 和"国王"很接近！
  "苹果" → [-0.5, 0.1, 0.9, 0.4, -0.2]  ← 和"国王"差很远

通过词向量，我们可以：
  1. 计算词与词的相似度（国王 ≈ 女王）
  2. 做类比推理（国王 - 男人 + 女人 ≈ 女王）
  3. 作为机器学习模型的输入特征

==============================================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from collections import defaultdict


# ==============================================================================
# 第一部分：词向量基础概念
# ==============================================================================
#
# 什么是词向量（Word Embedding）？
#
# 在传统 NLP 中，我们用 One-Hot 编码表示词：
#   "国王" → [1, 0, 0, 0, 0]
#   "女王" → [0, 1, 0, 0, 0]
#   "苹果" → [0, 0, 1, 0, 0]
#
# 问题：
#   1. 维度太高（词表有5万个词，每个词就是5万维向量）
#   2. 稀疏（大部分是0）
#   3. 无法表示语义关系（"国王"和"女王"的相似度 = 0）
#
# Word2Vec 的解决方案：
#   把每个词映射到一个低维稠密向量（如 100 维或 300 维）
#   "国王" → [0.2, 0.8, 0.1, ...]  （100维）
#   "女王" → [0.3, 0.7, 0.1, ...]  （100维，和"国王"接近！）
#
# ==============================================================================

def explain_one_hot():
    """
    演示 One-Hot 编码的问题

    One-Hot 编码：每个词用一个很长的向量表示，
    只有自己位置是1，其他都是0。
    """
    # 假设词表只有5个词
    vocab = ["国王", "女王", "男人", "女人", "苹果"]
    vocab_size = len(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    print("One-Hot 编码示例（词表大小 = {}）:".format(vocab_size))
    print("-" * 40)

    one_hot_vectors = {}
    for word in vocab:
        vec = np.zeros(vocab_size)
        vec[word_to_idx[word]] = 1
        one_hot_vectors[word] = vec
        print(f"  {word}: {vec.astype(int).tolist()}")

    # 计算相似度
    print("\n余弦相似度：")
    for w1, w2 in [("国王", "女王"), ("国王", "苹果"), ("女人", "女王")]:
        v1, v2 = one_hot_vectors[w1], one_hot_vectors[w2]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print(f"  {w1} vs {w2}: {cos_sim:.4f}")

    print("\n问题：所有词之间的相似度都是 0！One-Hot 无法捕捉语义关系。")

    return one_hot_vectors


# ==============================================================================
# 第二部分：Word2Vec 的两种模型
# ==============================================================================
#
# Word2Vec 有两种训练方式：
#
# 1. CBOW（Continuous Bag of Words）—— 用上下文预测中心词
#
#    生活类比：完形填空
#    "我 喜欢 ___ 苹果"
#    根据 "我"、"喜欢"、"苹果" 来猜测中间的词是"吃"
#
#    输入：上下文词的向量（平均或拼接）
#    输出：预测中心词
#
# 2. Skip-gram —— 用中心词预测上下文
#
#    生活类比：举一反三
#    看到"吃"这个词，能猜到它周围的词可能是：
#    "我"、"喜欢"、"苹果"、"面包" 等
#
#    输入：中心词的向量
#    输出：预测上下文词
#
# 一般来说：
#   - CBOW 训练速度快，适合高频词
#   - Skip-gram 效果好，适合低频词
#
# ==============================================================================

def build_vocabulary(corpus: list) -> tuple:
    """
    从语料库构建词表

    ━━━━━━━ 生活类比 ━━━━━━━
    就像统计一个城市有哪些居民：
    1. 走遍每条街道（遍历语料库）
    2. 记录遇到的每个人（统计词频）
    3. 给每个人编个号（分配索引）

    参数：
        corpus: 分词后的语料库，如 [["我", "喜欢", "吃"], ["他", "喜欢", "喝"]]

    返回：
        (词表列表, 词到索引的映射, 词频字典)
    """
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1

    # 按词频排序（高频词在前）
    vocab = sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    return vocab, word_to_idx, word_counts


def generate_training_data(corpus: list, word_to_idx: dict,
                           window_size: int = 2) -> list:
    """
    生成 CBOW / Skip-gram 的训练数据

    ━━━━━━━ 生活类比 ━━━━━━━
    就像出完形填空题：
    原句："我 喜欢 吃 苹果"
    窗口大小 = 2

    CBOW 训练样本（用上下文预测中心词）：
      上下文: ["我", "喜欢", "苹果"] → 预测: "吃"

    Skip-gram 训练样本（用中心词预测上下文）：
      中心词: "吃" → 预测: "我"
      中心词: "吃" → 预测: "喜欢"
      中心词: "吃" → 预测: "苹果"

    参数：
        corpus: 分词后的语料库
        word_to_idx: 词到索引的映射
        window_size: 上下文窗口大小

    返回：
        CBOW: [([上下文词索引], 中心词索引), ...]
        Skip-gram: [(中心词索引, 上下文词索引), ...]
    """
    cbow_data = []
    skipgram_data = []

    for sentence in corpus:
        # 把词转换为索引
        indices = [word_to_idx[w] for w in sentence if w in word_to_idx]

        for i, center_idx in enumerate(indices):
            # 获取上下文窗口内的词
            context_indices = []
            for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                if j != i:  # 排除中心词本身
                    context_indices.append(indices[j])

            if not context_indices:
                continue

            # CBOW: (上下文, 中心词)
            cbow_data.append((context_indices, center_idx))

            # Skip-gram: (中心词, 每个上下文词)
            for ctx_idx in context_indices:
                skipgram_data.append((center_idx, ctx_idx))

    return cbow_data, skipgram_data


# ==============================================================================
# 第三部分：简化版 Word2Vec（CBOW）实现
# ==============================================================================
#
# Word2Vec 的神经网络结构非常简单：
#
# CBOW 模型：
#
#   输入层（上下文词的One-Hot）     隐藏层（词向量）     输出层（预测词）
#   ┌─────────────────┐           ┌──────────┐        ┌─────────────┐
#   │ [1,0,0,0,0] (我) │──┐       │          │        │             │
#   │ [0,0,0,1,0] (吃) │──┼──→    │  W_input │  →     │   W_output  │ → softmax
#   │ [0,0,0,0,1] (苹果)│──┘       │          │        │             │
#   └─────────────────┘           └──────────┘        └─────────────┘
#                                      ↑
#                                   这就是词向量！
#
# 训练完成后，W_input 的每一行就是对应词的词向量。
#
# ==============================================================================

class SimpleWord2Vec:
    """
    简化版 Word2Vec（CBOW 模型）

    ━━━━━━━ 核心原理 ━━━━━━━

    1. 输入：上下文词的 One-Hot 向量
    2. 通过权重矩阵 W_input 得到隐藏层（词向量）
    3. 通过权重矩阵 W_output 得到输出层
    4. 用 Softmax 计算每个词作为中心词的概率
    5. 用交叉熵损失函数训练
    6. 训练完成后，W_input 就是词向量矩阵

    ━━━━━━━ 参数 ━━━━━━━
    vocab_size: 词表大小
    embedding_dim: 词向量维度（通常 50-300）
    learning_rate: 学习率
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 50,
                 learning_rate: float = 0.01):
        """
        初始化 Word2Vec 模型

        参数：
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            learning_rate: 学习率
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # 权重矩阵初始化（使用正态分布）
        # W_input: 输入层到隐藏层（这就是我们要的词向量矩阵！）
        self.W_input = np.random.randn(vocab_size, embedding_dim) * 0.01
        # W_output: 隐藏层到输出层
        self.W_output = np.random.randn(embedding_dim, vocab_size) * 0.01

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax 函数 —— 把任意实数转换为概率分布

        ━━━━━━━ 生活类比 ━━━━━━━
        就像考试成绩标准化：
        原始分数: [80, 90, 70]（无法直接比较占比）
        Softmax后: [0.24, 0.67, 0.09]（加起来=1，可以比较概率）
        """
        # 数值稳定性：减去最大值防止溢出
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, context_indices: list) -> np.ndarray:
        """
        前向传播（CBOW）

        ━━━━━━━ 步骤 ━━━━━━━
        1. 获取上下文词的向量
        2. 取平均得到隐藏层
        3. 乘以输出矩阵得到输出
        4. Softmax 得到概率分布

        参数：
            context_indices: 上下文词的索引列表

        返回：
            每个词作为中心词的概率
        """
        # 1. 获取上下文词的向量（查表）
        context_vectors = self.W_input[context_indices]  # (context_size, embedding_dim)

        # 2. 取平均（CBOW 的核心操作）
        hidden = np.mean(context_vectors, axis=0)  # (embedding_dim,)

        # 3. 计算输出
        output = hidden @ self.W_output  # (vocab_size,)

        # 4. Softmax
        probs = self._softmax(output)

        return probs, hidden

    def backward(self, context_indices: list, target_idx: int,
                 probs: np.ndarray, hidden: np.ndarray):
        """
        反向传播（更新权重）

        ━━━━━━━ 核心思想 ━━━━━━━
        用梯度下降法更新权重：
        1. 计算输出层误差（预测概率 - 真实标签）
        2. 误差反向传播到隐藏层
        3. 更新 W_output 和 W_input

        参数：
            context_indices: 上下文词的索引
            target_idx: 目标词（中心词）的索引
            probs: 前向传播输出的概率
            hidden: 隐藏层值
        """
        # 计算输出层误差
        output_error = probs.copy()
        output_error[target_idx] -= 1  # 交叉熵损失的梯度

        # 更新 W_output
        # dW_output = hidden^T * output_error
        self.W_output -= self.lr * np.outer(hidden, output_error)

        # 计算隐藏层误差
        hidden_error = self.W_output @ output_error  # (embedding_dim,)

        # 更新 W_input（上下文词的向量）
        for ctx_idx in context_indices:
            self.W_input[ctx_idx] -= self.lr * hidden_error

    def train(self, training_data: list, epochs: int = 100):
        """
        训练模型

        参数：
            training_data: [(上下文索引列表, 目标词索引), ...]
            epochs: 训练轮数
        """
        for epoch in range(epochs):
            total_loss = 0

            for context_indices, target_idx in training_data:
                # 前向传播
                probs, hidden = self.forward(context_indices)

                # 计算损失（交叉熵）
                loss = -np.log(probs[target_idx] + 1e-10)
                total_loss += loss

                # 反向传播
                self.backward(context_indices, target_idx, probs, hidden)

            # 每隔一定轮数打印损失
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(training_data)
                print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def get_word_vector(self, word_idx: int) -> np.ndarray:
        """
        获取词向量

        训练完成后，W_input 的每一行就是对应词的词向量
        """
        return self.W_input[word_idx]

    def most_similar(self, word_idx: int, word_to_idx: dict,
                     idx_to_word: dict, top_k: int = 5) -> list:
        """
        找最相似的词

        ━━━━━━━ 原理 ━━━━━━━
        用余弦相似度比较词向量，
        找到和目标词最接近的 top_k 个词。

        参数：
            word_idx: 目标词的索引
            word_to_idx: 词到索引的映射
            idx_to_word: 索引到词的映射
            top_k: 返回前k个

        返回：
            [(词, 相似度), ...]
        """
        target_vec = self.W_input[word_idx]
        similarities = []

        for i in range(self.vocab_size):
            if i == word_idx:
                continue
            vec = self.W_input[i]
            # 余弦相似度
            cos_sim = np.dot(target_vec, vec) / (
                np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10
            )
            similarities.append((idx_to_word[i], cos_sim))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# ==============================================================================
# 第四部分：Skip-gram 模型
# ==============================================================================

class SimpleSkipGram:
    """
    简化版 Skip-gram 模型

    ━━━━━━━ 与 CBOW 的区别 ━━━━━━━
    CBOW:     上下文 → 预测中心词（多个输入，一个输出）
    Skip-gram: 中心词 → 预测上下文（一个输入，多个输出）

    Skip-gram 的训练过程：
    对于每个 (中心词, 上下文词) 对：
    1. 输入中心词的 One-Hot
    2. 通过 W_input 得到隐藏层
    3. 通过 W_output 得到输出
    4. 用 Softmax 预测上下文词
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 50,
                 learning_rate: float = 0.01):
        """初始化 Skip-gram 模型"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # 权重矩阵
        self.W_input = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_output = np.random.randn(embedding_dim, vocab_size) * 0.01

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train_pair(self, center_idx: int, context_idx: int):
        """
        训练一个 (中心词, 上下文词) 对

        ━━━━━━━ 步骤 ━━━━━━━
        1. 前向传播：中心词 → 预测上下文词
        2. 计算损失
        3. 反向传播更新权重
        """
        # 前向传播
        hidden = self.W_input[center_idx]  # (embedding_dim,)
        output = hidden @ self.W_output    # (vocab_size,)
        probs = self._softmax(output)

        # 计算损失
        loss = -np.log(probs[context_idx] + 1e-10)

        # 反向传播
        output_error = probs.copy()
        output_error[context_idx] -= 1

        # 更新 W_output
        self.W_output -= self.lr * np.outer(hidden, output_error)

        # 更新 W_input
        hidden_error = self.W_output @ output_error
        self.W_input[center_idx] -= self.lr * hidden_error

        return loss

    def train(self, training_data: list, epochs: int = 100):
        """
        训练模型

        参数：
            training_data: [(中心词索引, 上下文词索引), ...]
            epochs: 训练轮数
        """
        for epoch in range(epochs):
            total_loss = 0
            for center_idx, context_idx in training_data:
                loss = self.train_pair(center_idx, context_idx)
                total_loss += loss

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(training_data)
                print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def get_word_vector(self, word_idx: int) -> np.ndarray:
        """获取词向量"""
        return self.W_input[word_idx]


# ==============================================================================
# 第五部分：词向量的神奇应用
# ==============================================================================

def word_analogy(model, word_to_idx, idx_to_word,
                 word_a: str, word_b: str, word_c: str, top_k: int = 3):
    """
    词类比推理：a - b + c = ?

    ━━━━━━━ 经典例子 ━━━━━━━
    国王 - 男人 + 女人 ≈ 女王
    北京 - 中国 + 日本 ≈ 东京

    ━━━━━━━ 原理 ━━━━━━━
    词向量可以捕捉语义关系：
    "国王" 和 "女王" 的向量差 ≈ "男人" 和 "女人" 的向量差
    （都代表了 "性别" 这个语义维度的变化）

    参数：
        model: 训练好的 Word2Vec 模型
        word_to_idx: 词到索引
        idx_to_word: 索引到词
        word_a, word_b, word_c: 类比词
        top_k: 返回前k个结果
    """
    if word_a not in word_to_idx or word_b not in word_to_idx or word_c not in word_to_idx:
        return []

    # 计算目标向量：a - b + c
    vec_a = model.W_input[word_to_idx[word_a]]
    vec_b = model.W_input[word_to_idx[word_b]]
    vec_c = model.W_input[word_to_idx[word_c]]
    target_vec = vec_a - vec_b + vec_c

    # 找最相似的词
    exclude = {word_to_idx.get(word_a, -1), word_to_idx.get(word_b, -1),
               word_to_idx.get(word_c, -1)}

    similarities = []
    for i in range(model.vocab_size):
        if i in exclude:
            continue
        vec = model.W_input[i]
        cos_sim = np.dot(target_vec, vec) / (
            np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10
        )
        similarities.append((idx_to_word[i], cos_sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ==============================================================================
# 第六部分：gensim Word2Vec 使用演示
# ==============================================================================

def demo_gensim_word2vec():
    """
    演示 gensim 库的 Word2Vec 用法

    gensim 是 Python 中最流行的 Word2Vec 实现库。
    在实际项目中，我们一般用 gensim 而不是自己实现。
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  [提示] gensim 未安装，请运行: pip install gensim")
        print("  以下为 gensim 用法说明（无需安装也能看懂）：")
        print("""
    ━━━━━━━ gensim Word2Vec 基本用法 ━━━━━━━

    from gensim.models import Word2Vec

    # 1. 准备语料库（分词后的句子列表）
    sentences = [
        ["我", "喜欢", "吃", "苹果"],
        ["他", "喜欢", "喝", "牛奶"],
        ["机器学习", "是", "人工智能", "的", "核心"],
    ]

    # 2. 训练模型
    model = Word2Vec(
        sentences,       # 语料库
        vector_size=100, # 词向量维度
        window=5,        # 上下文窗口大小
        min_count=1,     # 最小词频（低于此频率的词被忽略）
        sg=1,            # 0=CBOW, 1=Skip-gram
        epochs=100       # 训练轮数
    )

    # 3. 获取词向量
    vec = model.wv["机器学习"]  # 获取"机器学习"的词向量

    # 4. 找相似词
    similar = model.wv.most_similar("机器学习", topn=5)

    # 5. 词类比
    result = model.wv.most_similar(
        positive=["国王", "女人"],  # 加上这些词
        negative=["男人"],           # 减去这个词
        topn=3
    )

    # 6. 保存和加载模型
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")
        """)
        return

    # 准备语料库
    corpus = [
        ["机器学习", "是", "人工智能", "的", "核心", "技术"],
        ["深度学习", "是", "机器学习", "的", "一个", "分支"],
        ["自然语言处理", "是", "人工智能", "的", "重要", "应用"],
        ["计算机视觉", "是", "人工智能", "的", "另一个", "应用"],
        ["搜索引擎", "使用", "自然语言处理", "技术"],
        ["推荐系统", "使用", "机器学习", "算法"],
        ["语音识别", "使用", "深度学习", "模型"],
    ]

    print("  训练 gensim Word2Vec 模型...")
    model = Word2Vec(
        corpus,
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=200
    )

    # 展示词向量
    print("\n  词向量示例（前5维）：")
    for word in ["机器学习", "深度学习", "人工智能"]:
        if word in model.wv:
            vec = model.wv[word]
            print(f"    {word}: [{', '.join(f'{v:.3f}' for v in vec[:5])}...]")

    # 找相似词
    print("\n  与 '机器学习' 最相似的词：")
    if "机器学习" in model.wv:
        for word, sim in model.wv.most_similar("机器学习", topn=3):
            print(f"    {word}: {sim:.4f}")


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_word_vectors():
    """演示词向量基础"""
    print("=" * 60)
    print("词向量基础概念")
    print("=" * 60)

    explain_one_hot()

    print("\n\nWord2Vec 的解决方案：")
    print("  把每个词映射到低维稠密向量（如 50 维或 100 维）")
    print("  语义相近的词，向量也相近！")


def demo_cbow():
    """演示 CBOW 训练"""
    print("\n" + "=" * 60)
    print("CBOW 模型训练演示")
    print("=" * 60)

    # 准备语料
    corpus = [
        ["我", "喜欢", "吃", "苹果"],
        ["我", "喜欢", "喝", "牛奶"],
        ["他", "喜欢", "吃", "香蕉"],
        ["她", "喜欢", "喝", "果汁"],
        ["我", "不喜欢", "吃", "辣椒"],
    ]

    # 构建词表
    vocab, word_to_idx, word_counts = build_vocabulary(corpus)
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    print(f"\n  词表大小: {len(vocab)}")
    print(f"  词表: {vocab}")
    print(f"  词频: {dict(word_counts)}")

    # 生成训练数据
    cbow_data, _ = generate_training_data(corpus, word_to_idx, window_size=2)
    print(f"\n  CBOW 训练样本数: {len(cbow_data)}")
    print(f"  示例: 上下文 {cbow_data[0][0]} → 目标 {cbow_data[0][1]}")

    # 训练模型
    print("\n  开始训练 CBOW 模型...")
    model = SimpleWord2Vec(vocab_size=len(vocab), embedding_dim=50, learning_rate=0.01)
    model.train(cbow_data, epochs=100)

    # 展示词向量
    print("\n  训练完成！词向量示例（前5维）：")
    for word in ["喜欢", "吃", "喝"]:
        idx = word_to_idx[word]
        vec = model.get_word_vector(idx)
        print(f"    {word}: [{', '.join(f'{v:.3f}' for v in vec[:5])}...]")

    # 找相似词
    print("\n  与 '喜欢' 最相似的词：")
    idx = word_to_idx["喜欢"]
    similar = model.most_similar(idx, word_to_idx, idx_to_word, top_k=3)
    for word, sim in similar:
        print(f"    {word}: {sim:.4f}")


def demo_skipgram():
    """演示 Skip-gram 训练"""
    print("\n" + "=" * 60)
    print("Skip-gram 模型训练演示")
    print("=" * 60)

    corpus = [
        ["机器学习", "是", "人工智能", "核心"],
        ["深度学习", "是", "机器学习", "分支"],
        ["自然语言处理", "是", "人工智能", "应用"],
    ]

    vocab, word_to_idx, _ = build_vocabulary(corpus)
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    _, skipgram_data = generate_training_data(corpus, word_to_idx, window_size=2)
    print(f"\n  词表大小: {len(vocab)}")
    print(f"  Skip-gram 训练样本数: {len(skipgram_data)}")

    print("\n  开始训练 Skip-gram 模型...")
    model = SimpleSkipGram(vocab_size=len(vocab), embedding_dim=50, learning_rate=0.01)
    model.train(skipgram_data, epochs=100)

    print("\n  训练完成！词向量示例（前5维）：")
    for word in ["机器学习", "人工智能", "深度学习"]:
        if word in word_to_idx:
            idx = word_to_idx[word]
            vec = model.get_word_vector(idx)
            print(f"    {word}: [{', '.join(f'{v:.3f}' for v in vec[:5])}...]")


def demo_gensim():
    """演示 gensim Word2Vec"""
    print("\n" + "=" * 60)
    print("gensim Word2Vec 实战")
    print("=" * 60)
    demo_gensim_word2vec()


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十三章                      ║
    ║        Word2Vec 词向量                                ║
    ╚══════════════════════════════════════════════════════╝
    """)

    demo_word_vectors()
    demo_cbow()
    demo_skipgram()
    demo_gensim()

    # 总结
    print("\n" + "=" * 60)
    print("第十三章 总结")
    print("=" * 60)
    print("""
    本章我们学习了 Word2Vec 词向量：

    [OK] 词向量概念 — 给每个词一个"数字身份证"
    [OK] One-Hot 问题 — 高维稀疏、无法捕捉语义
    [OK] CBOW 模型 — 用上下文预测中心词（完形填空）
    [OK] Skip-gram 模型 — 用中心词预测上下文（举一反三）
    [OK] 词类比推理 — 国王 - 男人 + 女人 ≈ 女王
    [OK] gensim 实战 — 工业界的标准 Word2Vec 工具
    """)

    # =============================================
    # 下节课预告
    # =============================================
    """
    下节课我们将学习文本分类（Text Classification）：
    - 朴素贝叶斯分类器 —— 基于概率的经典分类算法
    - 支持向量机（SVM）—— 最大间隔分类器
    - sklearn 文本分类流水线 —— 从特征提取到模型预测的完整流程
    """
