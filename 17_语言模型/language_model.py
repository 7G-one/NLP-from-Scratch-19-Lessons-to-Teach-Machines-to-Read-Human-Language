"""
==============================================================================
第十七章：语言模型
==============================================================================
日期：2026-05-16

同学们好！这节课我们来学习语言模型 —— NLP 中最核心的概念之一。

----------------------------------------------------------------------
生活类比：语言模型就像"输入法的联想功能"
----------------------------------------------------------------------

当你在手机上打字时，输入法会"猜"你想打的下一个字：

  你输入：今
  输入法联想：天、年、晚、后...

  你输入：今天天
  输入法联想：气、真、不...

输入法为什么能"猜"得这么准？因为它背后有一个"语言模型"：
  - 它学习了大量文本数据
  - 知道"今天"后面跟"气"的概率很高
  - 知道"今天"后面跟"牛"的概率很低

语言模型的本质：
  给定前面的词，预测下一个词出现的概率。

----------------------------------------------------------------------
语言模型的两种类型
----------------------------------------------------------------------

1. N-Gram 模型（统计方法）
   → 看前 N-1 个词，统计第 N 个词的概率
   → 简单、快速，但需要大量数据

2. 神经网络语言模型（深度学习方法）
   → 用 RNN/LSTM/Transformer 学习语言规律
   → 更强大，但需要更多计算资源

----------------------------------------------------------------------
本章内容
----------------------------------------------------------------------

1. N-Gram 语言模型
2. 困惑度（Perplexity）评估
3. 文本生成
4. LSTM 语言模型概念

==============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import random
from collections import Counter, defaultdict


# ==============================================================================
# 第一部分：N-Gram 语言模型
# ==============================================================================
#
# N-Gram 是最简单的语言模型。
#
# 核心思想：
#   一个词出现的概率只和它前面的 N-1 个词有关。
#
# 生活类比：
#   想象你在玩"成语接龙"：
#   - 1-Gram（Unigram）：只看当前字 → "天" 出现的概率
#   - 2-Gram（Bigram）：看前 1 个字 → "今" 后面是 "天" 的概率
#   - 3-Gram（Trigram）：看前 2 个字 → "今天" 后面是 "气" 的概率
#
#   N 越大，考虑的上下文越多，但需要的数据也越多。
#
# ==============================================================================


class NGramModel:
    """
    N-Gram 语言模型

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你是一个老师，记录了学生们的作文：
    - "今天天气" 出现了 10 次
    - "今天天气好" 出现了 8 次
    - "今天天气坏" 出现了 2 次

    那么 "今天天气" 后面跟 "好" 的概率 = 8/10 = 0.8
    """

    def __init__(self, n: int = 2):
        """
        初始化 N-Gram 模型

        参数：
            n: N-Gram 的 N 值（默认 2，即 Bigram）
        """
        self.n = n
        # ngram_counts: 记录每个 N-Gram 出现的次数
        # 例如 Bigram: {("今", "天"): {"气": 5, "好": 3}}
        self.ngram_counts = defaultdict(Counter)
        # prefix_counts: 记录每个前缀出现的总次数
        self.prefix_counts = Counter()
        # vocabulary: 词汇表
        self.vocabulary = set()

    def tokenize(self, text: str) -> list:
        """
        简单的分词（按字符切分）

        参数：
            text: 输入文本

        返回：
            token 列表
        """
        tokens = []
        for char in text:
            if char.strip() and char not in "，。！？、；：""''（）【】《》\n\r\t":
                tokens.append(char)
        return tokens

    def train(self, texts: list):
        """
        训练 N-Gram 模型

        ━━━━━━━ 生活类比 ━━━━━━━
        就像一个老师批改大量作文，记录：
        - "今天" 后面跟 "气" 出现了多少次
        - "今天" 后面跟 "好" 出现了多少次
        - ...

        记录得越多，预测就越准。

        参数：
            texts: 训练文本列表
        """
        for text in texts:
            tokens = self.tokenize(text)

            # 添加开始和结束标记
            # <s> 表示句子开始，</s> 表示句子结束
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            # 更新词汇表
            self.vocabulary.update(tokens)

            # 统计 N-Gram
            for i in range(len(tokens) - self.n + 1):
                # 前缀：前 N-1 个词
                prefix = tuple(tokens[i:i + self.n - 1])
                # 目标：第 N 个词
                target = tokens[i + self.n - 1]

                self.ngram_counts[prefix][target] += 1
                self.prefix_counts[prefix] += 1

    def probability(self, prefix: tuple, word: str) -> float:
        """
        计算条件概率 P(word | prefix)

        ━━━━━━━ 生活类比 ━━━━━━━
        就像问：已知前面是"今天"，下一个字是"气"的概率是多少？

        公式：
            P(word | prefix) = count(prefix + word) / count(prefix)

        参数：
            prefix: 前缀（元组）
            word: 目标词

        返回：
            概率值（0 到 1）
        """
        prefix_count = self.prefix_counts.get(prefix, 0)
        if prefix_count == 0:
            # 未知前缀，返回均匀分布
            return 1.0 / len(self.vocabulary) if self.vocabulary else 0

        word_count = self.ngram_counts[prefix].get(word, 0)

        # 使用拉普拉斯平滑（+1 平滑）
        # 避免概率为 0
        return (word_count + 1) / (prefix_count + len(self.vocabulary))

    def predict_next(self, context: list, top_k: int = 5) -> list:
        """
        预测下一个词

        ━━━━━━━ 生活类比 ━━━━━━━
        就像输入法的联想功能：
        - 你输入"今天"
        - 输入法告诉你最可能的下一个字

        参数：
            context: 上下文（词列表）
            top_k: 返回前 k 个候选

        返回：
            候选词列表，每个元素是 (词, 概率)
        """
        # 取最后 N-1 个词作为前缀
        prefix = tuple(context[-(self.n - 1):]) if len(context) >= self.n - 1 else tuple(context)

        # 计算所有词的概率
        probs = []
        for word in self.vocabulary:
            if word in ('<s>', '</s>'):
                continue
            prob = self.probability(prefix, word)
            probs.append((word, prob))

        # 按概率降序排列
        probs.sort(key=lambda x: x[1], reverse=True)

        return probs[:top_k]

    def generate(self, max_length: int = 20, seed: list = None) -> str:
        """
        生成文本

        ━━━━━━━ 生活类比 ━━━━━━━
        就像让 AI "续写" 文章：
        1. 给它一个开头
        2. 它根据概率一个字一个字地往下写
        3. 直到写出结束标记或达到最大长度

        参数：
            max_length: 最大生成长度
            seed: 初始上下文（可选）

        返回：
            生成的文本
        """
        if seed is None:
            # 随机选择一个起始前缀
            prefixes = [p for p in self.ngram_counts.keys() if p[0] == '<s>']
            if not prefixes:
                return ""
            current = list(random.choice(prefixes))
        else:
            current = list(seed)

        result = list(current)

        for _ in range(max_length):
            # 取最后 N-1 个词作为前缀
            prefix = tuple(current[-(self.n - 1):]) if len(current) >= self.n - 1 else tuple(current)

            # 获取所有可能的下一个词及其概率
            candidates = []
            for word in self.vocabulary:
                if word in ('<s>',):
                    continue
                prob = self.probability(prefix, word)
                candidates.append((word, prob))

            if not candidates:
                break

            # 按概率随机选择（轮盘赌）
            total = sum(p for _, p in candidates)
            r = random.random() * total
            cumulative = 0
            next_word = candidates[-1][0]
            for word, prob in candidates:
                cumulative += prob
                if cumulative >= r:
                    next_word = word
                    break

            if next_word == '</s>':
                break

            result.append(next_word)
            current.append(next_word)

        return ''.join(result)


# ==============================================================================
# 第二部分：困惑度（Perplexity）
# ==============================================================================
#
# 困惑度是评估语言模型好坏的指标。
#
# 核心思想：
#   困惑度越低，模型预测越准。
#
# 生活类比：
#   想象你在玩"猜词游戏"：
#   - 如果模型很困惑（Perplexity 高）→ 猜很多次才猜对 → 模型不好
#   - 如果模型很确定（Perplexity 低）→ 一次就猜对 → 模型很好
#
# 公式：
#   PPL = 2^(-1/N * Σlog2(P(wi|w1...wi-1)))
#
# ==============================================================================


def calculate_perplexity(model: NGramModel, test_texts: list) -> float:
    """
    计算语言模型的困惑度（Perplexity）

    ━━━━━━━ 生活类比 ━━━━━━━
    困惑度就像"考试分数"（但越低越好）：
    - PPL = 1：完美预测（满分）
    - PPL = 100：平均每次有 100 个候选词（很困惑）
    - PPL = 1000：模型基本在瞎猜

    公式：
        PPL = 2^(-1/N * Σlog2(P(wi|context)))

    参数：
        model: 训练好的 N-Gram 模型
        test_texts: 测试文本列表

    返回：
        困惑度（越低越好）
    """
    total_log_prob = 0
    total_words = 0

    for text in test_texts:
        tokens = model.tokenize(text)
        tokens = ['<s>'] * (model.n - 1) + tokens + ['</s>']

        for i in range(model.n - 1, len(tokens)):
            prefix = tuple(tokens[i - model.n + 1:i])
            word = tokens[i]

            prob = model.probability(prefix, word)
            if prob > 0:
                total_log_prob += math.log2(prob)
            else:
                total_log_prob += math.log2(1e-10)  # 避免 log(0)

            total_words += 1

    if total_words == 0:
        return float('inf')

    # PPL = 2^(-平均对数概率)
    avg_log_prob = total_log_prob / total_words
    perplexity = 2 ** (-avg_log_prob)

    return perplexity


# ==============================================================================
# 第三部分：LSTM 语言模型概念
# ==============================================================================
#
# LSTM（Long Short-Term Memory）是一种特殊的循环神经网络（RNN）。
#
# 核心思想：
#   LSTM 能够"记住"长期的上下文信息，
#   而不像 N-Gram 只能看前 N-1 个词。
#
# 生活类比：
#   想象你在读一本小说：
#   - N-Gram 只记得前几页的内容
#   - LSTM 能记住整本书的情节
#   - 所以 LSTM 写的"续写"更连贯
#
# ==============================================================================


def lstm_language_model_concept():
    """
    LSTM 语言模型的概念说明

    ━━━━━━━ LSTM 的核心组件 ━━━━━━━

    1. 遗忘门（Forget Gate）：
       → 决定"忘记"哪些旧信息
       → 就像你读新章节时，忘记不重要的细节

    2. 输入门（Input Gate）：
       → 决定"记住"哪些新信息
       → 就像你在书上划重点

    3. 输出门（Output Gate）：
       → 决定"输出"哪些信息
       → 就像你写读书笔记时，选择写哪些内容

    ━━━━━━━ LSTM 语言模型的工作流程 ━━━━━━━

    输入序列：我 喜 欢 学 习

    步骤 1：输入"我" → LSTM 状态更新
    步骤 2：输入"喜" → 结合"我"的记忆，更新状态
    步骤 3：输入"欢" → 结合"我喜"的记忆，更新状态
    步骤 4：输入"学" → 结合"我喜欢"的记忆，更新状态
    步骤 5：输入"习" → 结合"我喜欢学"的记忆，预测下一个词

    最终：LSTM 输出下一个词的概率分布
    """
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    LSTM 语言模型                         │
    │                                                         │
    │   输入：  我 → 喜 → 欢 → 学 → 习 → ?                   │
    │           ↓    ↓    ↓    ↓    ↓                         │
    │   LSTM:  [记忆单元不断更新]                               │
    │           ↓    ↓    ↓    ↓    ↓                         │
    │   输出:   无   无   无   无   无  → "NLP"               │
    │                                                         │
    │   优点：                                                │
    │   - 能捕捉长距离依赖                                    │
    │   - 自动学习语言规律                                    │
    │   - 生成质量更高                                        │
    │                                                         │
    │   缺点：                                                │
    │   - 需要大量训练数据                                    │
    │   - 训练时间长                                          │
    │   - 需要 GPU 加速                                       │
    └─────────────────────────────────────────────────────────┘
    """)


def try_pytorch_lstm():
    """
    尝试使用 PyTorch 构建简单的 LSTM 语言模型

    ━━━━━━━ 注意 ━━━━━━━
    这个函数会尝试导入 PyTorch。
    如果没有安装 PyTorch，会打印概念说明。
    """
    try:
        import torch
        import torch.nn as nn

        class SimpleLSTMLanguageModel(nn.Module):
            """
            简单的 LSTM 语言模型

            ━━━━━━━ 结构 ━━━━━━━
            1. 词嵌入层：把词转换为向量
            2. LSTM 层：处理序列信息
            3. 全连接层：输出下一个词的概率
            """

            def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64):
                super().__init__()
                # 词嵌入：把词索引转换为稠密向量
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                # LSTM 层：处理序列
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                # 全连接层：输出词表大小的概率分布
                self.fc = nn.Linear(hidden_dim, vocab_size)

            def forward(self, x):
                """
                前向传播

                参数：
                    x: 输入序列（batch_size, seq_len）

                返回：
                    输出概率（batch_size, seq_len, vocab_size）
                """
                # 词嵌入
                embedded = self.embedding(x)  # (batch, seq, embed)
                # LSTM 处理
                lstm_out, _ = self.lstm(embedded)  # (batch, seq, hidden)
                # 全连接层
                output = self.fc(lstm_out)  # (batch, seq, vocab)
                return output

        # 演示：创建一个小型 LSTM 模型
        vocab_size = 100
        model = SimpleLSTMLanguageModel(vocab_size)

        # 创建示例输入
        x = torch.randint(0, vocab_size, (1, 5))  # batch=1, seq_len=5

        # 前向传播
        output = model(x)

        print("  PyTorch LSTM 语言模型演示:")
        print(f"    词汇表大小: {vocab_size}")
        print(f"    输入形状: {x.shape}")
        print(f"    输出形状: {output.shape}")
        print(f"    模型参数数量: {sum(p.numel() for p in model.parameters())}")

        return True

    except ImportError:
        print("  [提示] PyTorch 未安装，跳过 LSTM 代码演示")
        print("  安装方式: pip install torch")
        return False


# ==============================================================================
# 第四部分：带平滑的 N-Gram 模型
# ==============================================================================


class SmoothedNGramModel:
    """
    带多种平滑技术的 N-Gram 模型

    ━━━━━━━ 为什么要平滑？ ━━━━━━━
    如果某个 N-Gram 在训练数据中从未出现过，
    它的概率就是 0，这会导致整个句子的概率为 0。

    平滑技术就是给这些"没见过"的 N-Gram 一个小概率，
    而不是直接给 0。

    生活类比：
        就像一个餐厅，菜单上没有的菜不代表不能做，
        只是概率比较低而已。
    """

    def __init__(self, n: int = 2, smoothing: str = 'kneser_ney', discount: float = 0.75):
        """
        初始化

        参数：
            n: N-Gram 的 N 值
            smoothing: 平滑方法 ('laplace', 'kneser_ney', 'interpolation')
            discount: Kneser-Ney 的折扣值 d（默认 0.75）
        """
        self.n = n
        self.smoothing = smoothing
        self.discount = discount
        self.ngram_counts = defaultdict(Counter)
        self.prefix_counts = Counter()
        self.vocabulary = set()
        self.total_tokens = 0
        # Kneser-Ney 专用：记录每个词出现在多少种不同的上下文中
        # continuation_counts[word] = 有多少种不同的前缀后面跟着 word
        self.continuation_counts = Counter()
        # 记录有多少种不同的词跟在某个前缀后面
        self.prefix_unique_successors = Counter()

    def train(self, texts: list):
        """
        训练模型

        ━━━━━━━ Kneser-Ney 需要额外统计 ━━━━━━━
        除了常规的 n-gram 计数外，还需要统计：
        1. continuation_counts：每个词出现在多少种不同的前缀后面
        2. prefix_unique_successors：每个前缀后面有多少种不同的词

        这两个统计量用于计算"续接概率"（continuation probability）。
        """
        for text in texts:
            tokens = []
            for char in text:
                if char.strip() and char not in "，。！？、；：""''（）【】《》\n\r\t":
                    tokens.append(char)

            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            self.vocabulary.update(tokens)
            self.total_tokens += len(tokens)

            # 记录每个前缀有哪些不同的后继词（用于 Kneser-Ney）
            prefix_successors = defaultdict(set)

            for i in range(len(tokens) - self.n + 1):
                prefix = tuple(tokens[i:i + self.n - 1])
                target = tokens[i + self.n - 1]
                self.ngram_counts[prefix][target] += 1
                self.prefix_counts[prefix] += 1
                # 记录这个前缀的后继词
                prefix_successors[prefix].add(target)

            # 更新 continuation_counts：target 出现在多少种不同前缀后面
            for prefix, successors in prefix_successors.items():
                self.prefix_unique_successors[prefix] += len(successors)
                for s in successors:
                    self.continuation_counts[s] += 1

    def probability(self, prefix: tuple, word: str) -> float:
        """
        计算平滑后的概率

        平滑方法：
        1. Laplace（加 1 平滑）：最简单，给每个计数 +1
        2. Interpolation（插值平滑）：混合不同阶的 N-Gram
        3. Kneser-Ney：最常用，效果最好的平滑方法
        """
        vocab_size = len(self.vocabulary)
        if vocab_size == 0:
            return 0

        if self.smoothing == 'laplace':
            # Laplace 平滑：P = (count + 1) / (total + V)
            count = self.ngram_counts[prefix].get(word, 0)
            total = self.prefix_counts.get(prefix, 0)
            return (count + 1) / (total + vocab_size)

        elif self.smoothing == 'kneser_ney':
            return self.probability_kneser_ney(prefix, word)

        elif self.smoothing == 'interpolation':
            # 插值平滑：混合 unigram, bigram, trigram
            lambda1, lambda2, lambda3 = 0.1, 0.3, 0.6

            # Unigram 概率
            word_total = sum(c.get(word, 0) for c in self.ngram_counts.values())
            p_uni = (word_total + 1) / (self.total_tokens + vocab_size) if self.total_tokens > 0 else 1 / vocab_size

            # Bigram 概率
            if len(prefix) >= 1:
                short_prefix = prefix[-1:]
                p_bi = self.ngram_counts[short_prefix].get(word, 0) / self.prefix_counts.get(short_prefix, 1)
            else:
                p_bi = p_uni

            # Trigram 概率
            count = self.ngram_counts[prefix].get(word, 0)
            total = self.prefix_counts.get(prefix, 0)
            p_tri = count / total if total > 0 else p_bi

            return lambda1 * p_uni + lambda2 * p_bi + lambda3 * p_tri

        else:
            # 默认：简单加 1 平滑
            count = self.ngram_counts[prefix].get(word, 0)
            total = self.prefix_counts.get(prefix, 0)
            return (count + 1) / (total + vocab_size)

    def probability_kneser_ney(self, context: tuple, word: str) -> float:
        """
        Kneser-Ney 平滑的概率计算

        ━━━━━━━ Kneser-Ney 的核心思想 ━━━━━━━
        传统的平滑方法（如 Laplace）在处理低阶模型时，
        使用简单的 unigram 概率 P(word) 作为回退。

        但 Kneser-Ney 认为，低阶模型不应该用"词频"，
        而应该用"续接概率"（continuation probability）。

        ━━━━━━━ 生活类比 ━━━━━━━
        想象你要预测下一个词：

        传统方法（unigram）：
          "的" 出现频率很高 → P("的") 很大
          → 无论什么上下文，"的"的概率都很高

        Kneser-Ney（续接概率）：
          "的" 虽然频率高，但它几乎只跟在特定词后面
          → 续接概率 = "的"出现在多少种不同上下文中
          → 如果 "的" 只跟在少数几种词后面，续接概率就低

        这样，Kneser-Ney 能更好地利用"低阶模型"的信息。

        ━━━━━━━ 公式 ━━━━━━━
        对于 bigram（N=2）：
          P_KN(w | u) = max(count(u,w) - d, 0) / count(u)
                       + λ(u) * P_continuation(w)

        其中：
          d = 折扣值（默认 0.75），从高频 n-gram "借"一些概率给低频
          λ(u) = 归一化系数，确保概率之和为 1
          P_continuation(w) = w 的续接概率

        续接概率：
          P_continuation(w) = |{v: count(v,w) > 0}| / |{(v',w'): count(v',w') > 0}|
          即：w 前面有多少种不同的词 / 所有 bigram 的种类数

        参数：
            context: 上下文（前缀元组）
            word: 目标词

        返回：
            Kneser-Ney 平滑后的概率
        """
        d = self.discount  # 折扣值，通常取 0.75
        vocab_size = len(self.vocabulary)

        if vocab_size == 0:
            return 1.0 / max(vocab_size, 1)

        # ─── 处理 unigram 的情况 ───
        if len(context) == 0:
            # 对于 unigram，直接使用续接概率
            cont_total = sum(self.continuation_counts.values())
            if cont_total == 0:
                return 1.0 / vocab_size
            return self.continuation_counts.get(word, 0) / cont_total

        # ─── 计算高阶（n-gram）部分 ───
        # max(count(context, word) - d, 0) / count(context)
        count_cw = self.ngram_counts[context].get(word, 0)
        count_c = self.prefix_counts.get(context, 0)

        if count_c == 0:
            # 上下文未见过，回退到低阶
            shorter_context = context[1:] if len(context) > 1 else ()
            return self.probability_kneser_ney(shorter_context, word)

        # 折扣后的高阶概率
        p_higher = max(count_cw - d, 0) / count_c

        # ─── 计算归一化系数 λ(context) ───
        # λ(context) = d / count(context) * |{w: count(context, w) > 0}|
        # 即：折扣值除以上下文计数，乘以该上下文后面有多少种不同的词
        unique_successors = len(self.ngram_counts[context])
        lambda_val = (d / count_c) * unique_successors

        # ─── 计算低阶续接概率 ───
        # 对于 bigram，低阶是 unigram 的续接概率
        # 对于 trigram，低阶是 bigram 的 Kneser-Ney
        shorter_context = context[1:] if len(context) > 1 else ()
        p_lower = self.probability_kneser_ney(shorter_context, word)

        # ─── 最终概率 = 高阶 + λ * 低阶 ───
        return p_higher + lambda_val * p_lower

    def perplexity(self, test_texts: list) -> float:
        """
        计算困惑度

        ━━━━━━━ 生活类比 ━━━━━━━
        困惑度就像"考试时的犹豫程度"：
        - PPL = 1：完全不犹豫，一次就猜对
        - PPL = 10：在 10 个选项中犹豫
        - PPL = 100：在 100 个选项中瞎猜

        参数：
            test_texts: 测试文本列表

        返回：
            困惑度（越低越好）
        """
        total_log_prob = 0.0
        total_words = 0

        for text in test_texts:
            tokens = []
            for char in text:
                if char.strip() and char not in "，。！？、；：""''（）【】《》\n\r\t":
                    tokens.append(char)

            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            for i in range(self.n - 1, len(tokens)):
                prefix = tuple(tokens[i - self.n + 1:i])
                word = tokens[i]
                prob = self.probability(prefix, word)
                if prob > 0:
                    total_log_prob += math.log2(prob)
                else:
                    total_log_prob += math.log2(1e-10)
                total_words += 1

        if total_words == 0:
            return float('inf')

        avg_log_prob = total_log_prob / total_words
        return 2 ** (-avg_log_prob)


# ==============================================================================
# 第五部分：LSTM 语言模型（需要 PyTorch）
# ==============================================================================
#
# LSTM（Long Short-Term Memory）是一种能够捕捉长距离依赖的神经网络。
#
# 生活类比：
#   想象你在读一本小说：
#   - N-Gram 只记得前几页的内容（上下文窗口有限）
#   - LSTM 能记住整本书的情节（通过记忆单元传递信息）
#   - 所以 LSTM 写的"续写"更连贯、更有逻辑
#
# LSTM 的三个门：
#   1. 遗忘门：决定忘记哪些旧信息（像橡皮擦）
#   2. 输入门：决定记住哪些新信息（像记笔记）
#   3. 输出门：决定输出哪些信息（像写作业时选择写什么）
#
# ==============================================================================

# 尝试导入 PyTorch，如果未安装则跳过
_torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _torch_available = True
except ImportError:
    pass


if _torch_available:

    class LSTMLanguageModel(nn.Module):
        """
        基于 LSTM 的神经网络语言模型

        ━━━━━━━ 模型结构 ━━━━━━━
        ┌──────────────────────────────────────────────┐
        │  输入序列: "我 喜 欢 学 习"                     │
        │          ↓                                    │
        │  词嵌入层: 把每个字转换为稠密向量                │
        │          ↓                                    │
        │  LSTM 层: 逐个处理，积累上下文记忆              │
        │          ↓                                    │
        │  全连接层: 把记忆向量映射到词表大小              │
        │          ↓                                    │
        │  Softmax: 输出下一个字的概率分布                │
        └──────────────────────────────────────────────┘

        ━━━━━━━ 生活类比 ━━━━━━━
        就像一个学生在做"完形填空"：
        1. 读到每个字（输入）
        2. 在脑子里记住上下文（LSTM 记忆）
        3. 根据记忆猜测下一个字（输出）
        4. 和正确答案比较（计算损失）
        5. 调整自己的"猜测策略"（反向传播）

        属性：
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏层维度
            char_to_idx: 字符到索引的映射
            idx_to_char: 索引到字符的映射
        """

        def __init__(self, vocab_size: int, embed_dim: int = 64,
                     hidden_dim: int = 128):
            """
            初始化 LSTM 语言模型

            ━━━━━━━ 参数说明 ━━━━━━━
            vocab_size: 词汇表大小（总共多少个不同的字）
              → 例如中文常用字约 5000 个

            embed_dim: 词嵌入维度（每个字用多少维的向量表示）
              → 就像给每个字分配一个"身份证号码"
              → 维度越大，能表达的含义越丰富，但计算量也越大
              → 常用值：64, 128, 256, 300

            hidden_dim: LSTM 隐藏层维度（记忆的"容量"）
              → 就像学生的"短期记忆容量"
              → 容量越大，能记住的上下文越多
              → 常用值：128, 256, 512

            参数：
                vocab_size: 词汇表大小
                embed_dim: 词嵌入维度
                hidden_dim: LSTM 隐藏层维度
            """
            super(LSTMLanguageModel, self).__init__()

            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim

            # 字符到索引 / 索引到字符的映射
            self.char_to_idx = {}
            self.idx_to_char = {}

            # ─── 词嵌入层 ───
            # 把离散的字符索引转换为连续的稠密向量
            # 就像把"身份证号"转换为"个人详细档案"
            self.embedding = nn.Embedding(vocab_size, embed_dim)

            # ─── LSTM 层 ───
            # 处理序列信息，积累上下文记忆
            # batch_first=True 表示输入形状是 (batch, seq, feature)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

            # ─── 全连接层 ───
            # 把 LSTM 的隐藏状态映射到词表大小
            # 输出每个字作为下一个字的"得分"
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            """
            前向传播

            ━━━━━━━ 数据流 ━━━━━━━
            x (batch_size, seq_len)
              → embedding: (batch_size, seq_len, embed_dim)
              → lstm: (batch_size, seq_len, hidden_dim)
              → fc: (batch_size, seq_len, vocab_size)

            参数：
                x: 输入序列，形状 (batch_size, seq_len)
                   每个元素是一个字符的索引

            返回：
                输出 logits，形状 (batch_size, seq_len, vocab_size)
                表示每个位置上每个字的"得分"
            """
            # 第一步：词嵌入
            # 把字符索引转换为稠密向量
            embedded = self.embedding(x)  # (batch, seq, embed)

            # 第二步：LSTM 处理
            # LSTM 逐个处理序列中的每个位置
            # output 包含每个位置的隐藏状态
            lstm_out, _ = self.lstm(embedded)  # (batch, seq, hidden)

            # 第三步：全连接层
            # 把隐藏状态映射到词表大小的 logits
            output = self.fc(lstm_out)  # (batch, seq, vocab)

            return output

        def build_vocab(self, texts: list):
            """
            从文本构建词汇表

            ━━━━━━━ 生活类比 ━━━━━━━
            就像老师在开学时统计"本学期会用到的所有字"，
            然后给每个字编一个学号。

            参数：
                texts: 文本列表
            """
            chars = set()
            for text in texts:
                for char in text:
                    if char.strip() and char not in "，。！？、；：""''（）【】《》\n\r\t":
                        chars.add(char)

            # 添加特殊标记
            chars.add('<s>')  # 句子开始
            chars.add('</s>')  # 句子结束
            chars.add('<unk>')  # 未知字符

            sorted_chars = sorted(chars)
            self.char_to_idx = {c: i for i, c in enumerate(sorted_chars)}
            self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

            # 更新词汇表大小
            self.vocab_size = len(sorted_chars)
            # 重新创建嵌入层和全连接层
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.fc = nn.Linear(self.hidden_dim, self.vocab_size)

        def train_model(self, corpus: list, epochs: int = 10,
                        learning_rate: float = 0.001):
            """
            训练 LSTM 语言模型

            ━━━━━━━ 训练流程 ━━━━━━━
            1. 把文本转换为数字序列
            2. 创建输入-目标对（输入是前 N 个字，目标是第 N+1 个字）
            3. 前向传播：用模型预测下一个字
            4. 计算损失：预测和真实答案的差距
            5. 反向传播：调整模型参数
            6. 重复多轮（epoch）

            ━━━━━━━ 生活类比 ━━━━━━━
            就像学生做练习题：
            1. 看题目（输入序列）
            2. 猜答案（前向传播）
            3. 对答案（计算损失）
            4. 找出错误原因（反向传播）
            5. 调整学习方法（更新参数）
            6. 做更多练习（下一个 epoch）

            参数：
                corpus: 训练语料（文本列表）
                epochs: 训练轮数
                learning_rate: 学习率（步长）
            """
            # 第一步：构建词汇表
            self.build_vocab(corpus)
            print(f"    词汇表大小: {self.vocab_size}")

            # 第二步：准备训练数据
            # 把所有文本转换为索引序列
            all_indices = []
            for text in corpus:
                for char in text:
                    if char.strip() and char not in "，。！？、；：""''（）【】《》\n\r\t":
                        idx = self.char_to_idx.get(char,
                                                    self.char_to_idx.get('<unk>', 0))
                        all_indices.append(idx)
                # 添加句子结束标记
                all_indices.append(self.char_to_idx.get('</s>', 0))

            if len(all_indices) < 2:
                print("    训练数据不足！")
                return

            # 第三步：创建输入-目标对
            # 输入：前 seq_len 个字
            # 目标：后 seq_len 个字（向右移一位）
            seq_len = 20  # 序列长度
            inputs = []
            targets = []

            for i in range(0, len(all_indices) - seq_len, seq_len):
                inputs.append(all_indices[i:i + seq_len])
                targets.append(all_indices[i + 1:i + seq_len + 1])

            if not inputs:
                # 数据太少，用更短的序列
                seq_len = max(2, len(all_indices) - 1)
                inputs = [all_indices[:seq_len]]
                targets = [all_indices[1:seq_len + 1]]

            # 转换为 PyTorch 张量
            inputs_tensor = torch.tensor(inputs, dtype=torch.long)
            targets_tensor = torch.tensor(targets, dtype=torch.long)

            print(f"    训练样本数: {len(inputs)}")
            print(f"    序列长度: {seq_len}")

            # 第四步：设置优化器和损失函数
            # Adam 优化器：自适应学习率，收敛快
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            # CrossEntropyLoss：分类问题的标准损失函数
            criterion = nn.CrossEntropyLoss()

            # 第五步：开始训练
            print(f"    开始训练...")
            self.train()  # 设置为训练模式

            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0

                # 小批量训练
                batch_size = 32
                for start in range(0, len(inputs_tensor), batch_size):
                    end = min(start + batch_size, len(inputs_tensor))
                    batch_inputs = inputs_tensor[start:end]
                    batch_targets = targets_tensor[start:end]

                    # 前向传播
                    optimizer.zero_grad()
                    output = self.forward(batch_inputs)

                    # 计算损失
                    # output: (batch, seq, vocab) → (batch*seq, vocab)
                    # targets: (batch, seq) → (batch*seq)
                    loss = criterion(output.reshape(-1, self.vocab_size),
                                     batch_targets.reshape(-1))

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪：防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    # 更新参数
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / max(num_batches, 1)
                if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
                    print(f"    Epoch {epoch + 1}/{epochs}: "
                          f"平均损失 = {avg_loss:.4f}")

            print("    训练完成！")

        def generate(self, start_text: str, length: int = 20,
                     temperature: float = 0.8) -> str:
            """
            生成文本

            ━━━━━━━ 生成过程 ━━━━━━━
            1. 从起始文本开始
            2. 把当前文本输入模型
            3. 模型输出下一个字的概率分布
            4. 根据概率分布采样一个字
            5. 把采样的字加到文本末尾
            6. 重复 2-5，直到达到指定长度

            ━━━━━━━ 温度参数 ━━━━━━━
            temperature 控制生成的"创造性"：
            - temperature < 1.0：更保守，倾向于选择高概率的字
            - temperature = 1.0：正常采样
            - temperature > 1.0：更有创造性，低概率的字也有机会被选中

            就像做选择题：
            - 温度低：只选最有把握的答案
            - 温度高：有时候也选不太有把握的答案

            参数：
                start_text: 起始文本（种子）
                length: 生成的最大长度
                temperature: 温度参数

            返回：
                生成的文本
            """
            self.eval()  # 设置为评估模式

            # 把起始文本转换为索引
            indices = []
            for char in start_text:
                idx = self.char_to_idx.get(char,
                                            self.char_to_idx.get('<unk>', 0))
                indices.append(idx)

            generated = list(indices)

            with torch.no_grad():  # 不需要计算梯度
                for _ in range(length):
                    # 准备输入
                    input_tensor = torch.tensor([generated], dtype=torch.long)

                    # 前向传播
                    output = self.forward(input_tensor)

                    # 取最后一个位置的输出
                    logits = output[0, -1, :] / temperature

                    # Softmax 转换为概率
                    probs = torch.softmax(logits, dim=0)

                    # 按概率分布采样
                    # multinomial 从分布中随机抽取一个样本
                    next_idx = torch.multinomial(probs, 1).item()

                    # 检查是否生成了结束标记
                    if next_idx == self.char_to_idx.get('</s>', -1):
                        break

                    generated.append(next_idx)

            # 把索引转换回字符
            result = ""
            for idx in generated:
                char = self.idx_to_char.get(idx, '<unk>')
                if char not in ('<s>', '</s>', '<unk>'):
                    result += char

            return result


def demo_lstm_language_model():
    """
    演示 LSTM 语言模型的训练和生成

    ━━━━━━━ 演示内容 ━━━━━━━
    1. 构建小型语料库
    2. 训练 LSTM 语言模型
    3. 用训练好的模型生成文本
    """
    if not _torch_available:
        print("  [提示] PyTorch 未安装，跳过 LSTM 语言模型演示")
        print("  安装方式: pip install torch")
        return False

    print("=" * 60)
    print("LSTM 语言模型训练演示")
    print("=" * 60)

    print("""
    ━━━━━━━ 训练流程 ━━━━━━━

    1. 准备语料：收集大量文本数据
    2. 构建词汇表：统计所有出现的字符
    3. 数字化：把字符转换为数字索引
    4. 创建训练样本：(前文, 下一个字) 的配对
    5. 训练模型：用 CrossEntropyLoss + Adam
    6. 生成文本：用训练好的模型续写

    ━━━━━━━ 关键概念 ━━━━━━━

    - 词嵌入（Embedding）：
      把离散的字符映射到连续的向量空间
      语义相近的字，向量距离也近

    - LSTM（长短期记忆）：
      一种特殊的 RNN，能捕捉长距离依赖
      通过三个"门"控制信息的遗忘、输入和输出

    - 损失函数（CrossEntropyLoss）：
      衡量模型预测和真实答案的差距
      差距越大，损失越大

    - 优化器（Adam）：
      自适应调整学习率的优化算法
      比普通 SGD 收敛更快
    """)

    # 构建语料
    corpus = [
        "今天天气真好适合出去玩",
        "今天天气不错心情好",
        "明天天气也很好",
        "好天气让人开心",
        "天气好出去走走",
        "今天学习自然语言处理",
        "自然语言处理很有趣",
        "学习语言模型很重要",
        "深度学习改变了人工智能",
        "人工智能是未来的发展方向",
        "机器学习是人工智能的核心",
        "自然语言处理需要大量数据",
        "语言模型可以生成文本",
        "LSTM能够捕捉长距离依赖",
        "神经网络模拟人脑的工作方式",
        "深度学习需要大量计算资源",
        "GPU加速了深度学习的训练",
        "预训练模型是NLP的突破",
        "Transformer架构改变了NLP",
        "注意力机制是Transformer的核心",
    ]

    # 创建模型
    model = LSTMLanguageModel(vocab_size=100, embed_dim=32, hidden_dim=64)

    # 训练模型
    print("  训练 LSTM 语言模型:")
    model.train_model(corpus, epochs=15, learning_rate=0.005)

    # 生成文本
    print("\n  生成文本示例:")
    seeds = ["今天", "自然", "深度", "人工"]
    for seed in seeds:
        text = model.generate(seed, length=15, temperature=0.7)
        print(f"    种子 '{seed}' → {text}")

    print("""
    ━━━━━━━ 分析 ━━━━━━━

    训练过程中的损失变化：
    - 刚开始损失很大（模型在瞎猜）
    - 随着训练进行，损失逐渐下降
    - 最终趋于稳定（模型学会了语言规律）

    生成文本的特点：
    - 短期内比较流畅（LSTM 记住了局部规律）
    - 长期可能不太连贯（训练数据太少）
    - 温度参数控制创造性 vs 确定性

    改进方向：
    1. 更多训练数据 → 生成更流畅
    2. 更大的模型 → 捕捉更复杂的规律
    3. 更长的训练 → 更好的收敛
    4. 使用 Transformer → 更好的长距离依赖建模
    """)

    return True


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十七章                      ║
    ║        语言模型                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 训练语料
    corpus = [
        "今天天气真好",
        "今天天气不错",
        "今天是好天气",
        "明天天气也很好",
        "天气好心情就好",
        "好天气适合出去玩",
        "今天适合出去走走",
        "明天适合出去玩",
    ]

    # 测试 N-Gram 模型
    print("=" * 60)
    print("  1. N-Gram 语言模型")
    print("=" * 60)

    for n in [2, 3]:
        print(f"\n  {n}-Gram 模型:")
        model = NGramModel(n=n)
        model.train(corpus)
        print(f"    词汇表大小: {len(model.vocabulary)}")
        print(f"    N-Gram 数量: {sum(len(v) for v in model.ngram_counts.values())}")

        # 预测下一个词
        context = ['今', '天']
        predictions = model.predict_next(context, top_k=3)
        print(f"    预测 '{''.join(context)}' 的下一个词:")
        for word, prob in predictions:
            print(f"      '{word}': {prob:.4f}")

        # 生成文本
        print(f"    生成文本: {model.generate(max_length=10)}")

    # 测试困惑度
    print("\n" + "=" * 60)
    print("  2. 困惑度（Perplexity）")
    print("=" * 60)

    model = NGramModel(n=2)
    model.train(corpus)

    test_texts = ["今天天气好", "明天出去玩"]
    ppl = calculate_perplexity(model, test_texts)
    print(f"  测试文本: {test_texts}")
    print(f"  困惑度: {ppl:.2f}（越低越好）")

    # LSTM 概念
    print("\n" + "=" * 60)
    print("  3. LSTM 语言模型概念")
    print("=" * 60)
    lstm_language_model_concept()

    # PyTorch LSTM 演示
    print("=" * 60)
    print("  4. PyTorch LSTM 演示")
    print("=" * 60)
    try_pytorch_lstm()

    # =============================================
    # 课程总结
    # =============================================
    """
    核心收获：
    - 语言模型的本质是预测下一个词的概率 —— N-Gram 通过统计历史出现频率来估计
    - 困惑度（Perplexity）是评估语言模型的核心指标 —— 越低说明模型预测越准
    - LSTM 通过遗忘门、输入门、输出门三个"阀门"控制信息流动 —— 解决了 RNN 的长期依赖问题

    常见陷阱：
    - 忽视平滑处理 —— 未见过的 N-Gram 概率为 0 会导致整个句子概率归零，必须做拉平滑
    - N-Gram 的 N 值过大 —— 上下文窗口越大需要的训练数据越多，数据稀疏问题越严重
    - LSTM 生成文本时温度参数设置不当 —— 温度过低导致重复，温度过高导致胡言乱语

    下节课预告：
    - 下一章我们将学习深度学习基础 —— 从感知机到 CNN、RNN、LSTM 的完整体系
    """
