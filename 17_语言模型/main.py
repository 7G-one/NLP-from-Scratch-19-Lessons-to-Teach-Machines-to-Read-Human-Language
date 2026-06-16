import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十七章：语言模型 — 完整演示
==============================================================================
G-one NLP 学院
日期：2026-05-16

运行方式：
    python main.py

前置知识：
    - Python 基础（列表、字典、类）
    - 概率论基础（条件概率）

本章内容：
    1. N-Gram 语言模型的构建与训练
    2. 文本生成
    3. 困惑度评估
    4. LSTM 语言模型概念
==============================================================================
"""

from language_model import (
    NGramModel,
    SmoothedNGramModel,
    calculate_perplexity,
    lstm_language_model_concept,
    try_pytorch_lstm,
    demo_lstm_language_model,
)

# LSTMLanguageModel 只在 PyTorch 可用时才存在
try:
    from language_model import LSTMLanguageModel
except ImportError:
    LSTMLanguageModel = None


def print_separator(title: str):
    """打印分隔线和标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def lesson_what_is_lm():
    """第一部分：什么是语言模型"""

    print_separator("17.1 什么是语言模型？")

    print("""
    ┌─────────────────────────────────────────────────────────┐
    │              语言模型 = 预测下一个词                       │
    │                                                         │
    │   核心问题：给定前面的词，下一个词是什么？                 │
    │                                                         │
    │   数学表达：P(wn | w1, w2, ..., wn-1)                   │
    │                                                         │
    │   例如：                                                │
    │     P("气" | "今天") = 0.6                               │
    │     P("好" | "今天") = 0.1                               │
    │     P("牛" | "今天") = 0.001                             │
    │                                                         │
    │   应用：                                                │
    │   - 输入法联想                                          │
    │   - 语音识别                                            │
    │   - 机器翻译                                            │
    │   - 文本生成                                            │
    │   - 拼写纠错                                            │
    └─────────────────────────────────────────────────────────┘
    """)


def lesson_ngram():
    """第二部分：N-Gram 模型"""

    print_separator("17.2 N-Gram 语言模型")

    print("""
    N-Gram 是最简单的语言模型：

    ┌─────────────────────────────────────────────────────────┐
    │  N 值    │  名称    │  考虑的上下文                      │
    ├─────────┼─────────┼───────────────────────────────────│
    │  N=1    │ Unigram  │ 不看上下文，只看词频               │
    │  N=2    │ Bigram   │ 看前 1 个词                        │
    │  N=3    │ Trigram  │ 看前 2 个词                        │
    │  N=4    │ 4-Gram   │ 看前 3 个词                        │
    └─────────┴─────────┴───────────────────────────────────┘

    优缺点：
    ┌──────────────┬─────────────────────────────────────────┐
    │  优点         │  简单、快速、易于理解                    │
    │  缺点         │  N 太小：上下文不足                      │
    │               │  N 太大：数据稀疏、计算量大              │
    └──────────────┴─────────────────────────────────────────┘
    """)

    # 训练语料
    corpus = [
        "今天天气真好适合出去玩",
        "今天天气不错心情好",
        "明天天气也很好",
        "好天气让人开心",
        "天气好出去走走",
        "今天学习自然语言处理",
        "自然语言处理很有趣",
        "学习语言模型很重要",
    ]

    print("  训练语料:")
    for doc in corpus:
        print(f"    - {doc}")

    # 训练 Bigram 模型
    print("\n  训练 Bigram 模型:")
    model = NGramModel(n=2)
    model.train(corpus)
    print(f"    词汇表大小: {len(model.vocabulary)}")

    # 预测演示
    print("\n  预测演示:")
    contexts = [['今', '天'], ['天', '气'], ['好']]
    for ctx in contexts:
        predictions = model.predict_next(ctx, top_k=3)
        pred_str = ", ".join([f"'{w}'({p:.3f})" for w, p in predictions])
        print(f"    '{''.join(ctx)}' → {pred_str}")


def lesson_text_generation():
    """第三部分：文本生成"""

    print_separator("17.3 文本生成")

    print("""
    语言模型最有趣的应用之一就是"文本生成"：

    ┌─────────────────────────────────────────────────────────┐
    │  步骤：                                                 │
    │  1. 给一个开头（种子）                                   │
    │  2. 预测下一个词                                         │
    │  3. 把预测的词加到序列末尾                               │
    │  4. 重复 2-3，直到达到指定长度                           │
    │                                                         │
    │  注意：每次预测时可以"随机采样"                          │
    │  → 同样的开头，可能生成不同的文本                        │
    └─────────────────────────────────────────────────────────┘
    """)

    corpus = [
        "今天天气真好适合出去玩",
        "今天天气不错心情好",
        "明天天气也很好",
        "好天气让人开心",
        "天气好出去走走",
        "今天学习自然语言处理",
        "自然语言处理很有趣",
        "学习语言模型很重要",
    ]

    model = NGramModel(n=2)
    model.train(corpus)

    print("  生成示例（Bigram 模型）:")
    for i in range(5):
        text = model.generate(max_length=8)
        print(f"    {i+1}. {text}")


def lesson_perplexity():
    """第四部分：困惑度"""

    print_separator("17.4 困惑度（Perplexity）")

    print("""
    困惑度是评估语言模型好坏的指标：

    ┌─────────────────────────────────────────────────────────┐
    │  困惑度 = 模型对文本的"惊讶程度"                         │
    │                                                         │
    │  - PPL = 1：完全不惊讶（完美预测）                       │
    │  - PPL = 100：有点惊讶（有 100 个候选词）                │
    │  - PPL = 1000：非常惊讶（模型在瞎猜）                    │
    │                                                         │
    │  公式：PPL = 2^(-1/N * Σlog2(P(wi|context)))            │
    │                                                         │
    │  直觉理解：                                             │
    │  困惑度 = 模型在每个位置平均有多少个"等概率的候选词"     │
    └─────────────────────────────────────────────────────────┘
    """)

    corpus = [
        "今天天气真好",
        "今天天气不错",
        "明天天气也很好",
        "好天气让人开心",
    ]

    model = NGramModel(n=2)
    model.train(corpus)

    test_cases = [
        "今天天气好",      # 和训练数据相似
        "今天天气真好",    # 和训练数据很相似
        "牛蛙苹果手机",    # 和训练数据完全不同
    ]

    print("  困惑度测试:")
    for text in test_cases:
        ppl = calculate_perplexity(model, [text])
        print(f"    '{text}' → PPL = {ppl:.2f}")

    print("\n  分析:")
    print("    - 和训练数据越相似的文本，困惑度越低")
    print("    - 和训练数据不同的文本，困惑度越高")


def lesson_lstm():
    """第五部分：LSTM 语言模型"""

    print_separator("17.5 LSTM 语言模型")

    lstm_language_model_concept()

    print("  PyTorch LSTM 演示:")
    try_pytorch_lstm()


def lesson_smoothing():
    """第六部分：平滑技术"""

    print_separator("17.6 平滑技术")

    print("""
    为什么要平滑？
    ┌─────────────────────────────────────────────────────────┐
    │  如果某个 N-Gram 在训练数据中从未出现，                  │
    │  它的概率就是 0，这会导致整个句子的概率为 0。            │
    │                                                         │
    │  平滑技术给这些"没见过"的 N-Gram 一个小概率。            │
    │                                                         │
    │  常见方法：                                             │
    │  1. Laplace（加 1 平滑）：最简单                        │
    │  2. Interpolation（插值）：混合不同阶 N-Gram            │
    │  3. Kneser-Ney：最常用，效果最好                        │
    └─────────────────────────────────────────────────────────┘
    """)

    corpus = [
        "今天天气真好",
        "今天天气不错",
        "明天天气也很好",
    ]

    print("  平滑方法对比:")
    for method in ['laplace', 'interpolation']:
        model = SmoothedNGramModel(n=2, smoothing=method)
        model.train(corpus)

        prefix = ('今', '天')
        test_words = ['气', '好', '牛']
        print(f"\n    {method}:")
        for word in test_words:
            prob = model.probability(prefix, word)
            print(f"      P('{word}' | '今天') = {prob:.4f}")


def lesson_kneser_ney():
    """
    第七部分：Kneser-Ney 平滑

    ━━━━━━━ 本节要点 ━━━━━━━
    1. Kneser-Ney 的核心思想：续接概率
    2. 绝对折扣法：从高频 n-gram 借概率给低频
    3. 与 Laplace 和插值平滑的对比
    """

    print_separator("17.7 Kneser-Ney 平滑")

    print("""
    Kneser-Ney 是目前最常用的平滑方法，效果优于 Laplace 和插值。

    ━━━━━━━ 核心思想 ━━━━━━━
    传统方法在回退到低阶模型时，使用简单的词频（unigram 概率）。
    但 Kneser-Ney 使用"续接概率"（continuation probability）。

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你要预测"___ 好"中空格处的字：

    传统方法（unigram）：
      "天" 出现频率很高 → P("天") 很大
      → 不管上下文是什么，"天"的概率都很高

    Kneser-Ney（续接概率）：
      "很" 虽然频率不算最高，但它能和很多词搭配
      → "很好"、"很棒"、"很快"、"很喜欢"...
      → 续接概率高 → 更可能出现在各种上下文中

    ━━━━━━━ 公式 ━━━━━━━
    P_KN(w | u) = max(count(u,w) - d, 0) / count(u)
                 + λ(u) * P_continuation(w)

    其中：
      d = 折扣值（0.75），从高频 n-gram 借出的概率
      λ(u) = 归一化系数
      P_continuation(w) = w 的续接概率
    """)

    corpus = [
        "今天天气真好",
        "今天天气不错",
        "明天天气也很好",
        "好天气让人开心",
        "天气好出去走走",
        "今天学习自然语言处理",
        "自然语言处理很有趣",
        "学习语言模型很重要",
    ]

    # 训练 Kneser-Ney 模型
    print("  训练 Kneser-Ney 平滑的 Bigram 模型:")
    model = SmoothedNGramModel(n=2, smoothing='kneser_ney', discount=0.75)
    model.train(corpus)
    print(f"    词汇表大小: {len(model.vocabulary)}")

    # 展示续接概率
    print("\n  续接概率（Continuation Probability）:")
    print("  续接概率 = 一个词出现在多少种不同上下文中")
    print(f"  {'词':<6} {'续接次数':<10} {'说明'}")
    print("  " + "-" * 40)

    # 按续接次数排序
    sorted_words = sorted(model.continuation_counts.items(),
                          key=lambda x: x[1], reverse=True)
    for word, count in sorted_words[:10]:
        if word in ('<s>', '</s>'):
            continue
        note = "上下文多样" if count > 3 else "上下文较少"
        print(f"  {word:<6} {count:<10} {note}")

    # 三种平滑方法对比
    print("\n  三种平滑方法对比:")
    print(f"  P(word | '今天'):")
    print(f"  {'方法':<15} {'气':<10} {'好':<10} {'牛':<10}")
    print("  " + "-" * 45)

    prefix = ('今', '天')
    test_words = ['气', '好', '牛']

    for method in ['laplace', 'interpolation', 'kneser_ney']:
        m = SmoothedNGramModel(n=2, smoothing=method)
        m.train(corpus)
        probs = []
        for w in test_words:
            p = m.probability(prefix, w)
            probs.append(f"{p:.4f}")
        print(f"  {method:<15} {probs[0]:<10} {probs[1]:<10} {probs[2]:<10}")

    # 困惑度对比
    print("\n  困惑度对比（越低越好）:")
    test_texts = ["今天天气好", "明天出去玩", "学习语言处理"]

    for method in ['laplace', 'interpolation', 'kneser_ney']:
        m = SmoothedNGramModel(n=2, smoothing=method)
        m.train(corpus)
        ppl = m.perplexity(test_texts)
        print(f"    {method:<15} PPL = {ppl:.2f}")

    print("""
    ━━━━━━━ 分析 ━━━━━━━

    Kneser-Ney 的优势：
    1. 续接概率比词频更能反映词的"通用性"
    2. 折扣法合理分配概率：从高频借给低频
    3. 归一化系数确保概率之和为 1

    为什么续接概率更好？
    - "的" 的词频很高，但它只跟在特定词后面
    - "好" 的词频可能不如 "的"，但它能和很多词搭配
    - 续接概率能区分这两种情况

    Kneser-Ney 的变体：
    - Modified Kneser-Ney：使用不同的折扣值（d1, d2, d3）
    - 高阶 Kneser-Ney：递归地使用续接概率
    """)


def lesson_lstm_training():
    """
    第八部分：LSTM 语言模型训练实战

    ━━━━━━━ 本节要点 ━━━━━━━
    1. LSTM 语言模型的构建
    2. 训练流程：前向传播 → 计算损失 → 反向传播
    3. 文本生成：温度参数的影响
    """

    print_separator("17.8 LSTM 语言模型训练实战")

    print("""
    前面我们了解了 LSTM 的概念，现在动手训练一个真正的 LSTM 语言模型！

    ━━━━━━━ 训练流程 ━━━━━━━
    ┌──────────────────────────────────────────────────┐
    │  语料 → 构建词汇表 → 数字化 → 创建训练样本        │
    │  → 前向传播 → 计算损失 → 反向传播 → 更新参数      │
    │  → 重复多轮 → 用模型生成文本                      │
    └──────────────────────────────────────────────────┘

    ━━━━━━━ 关键组件 ━━━━━━━
    - Embedding：字符 → 向量（像给每个字编身份证号）
    - LSTM：处理序列，积累上下文记忆
    - Linear：隐藏状态 → 词表大小的 logits
    - CrossEntropyLoss：分类损失
    - Adam：自适应学习率优化器
    """)

    demo_lstm_language_model()

    print("""
    ━━━━━━━ 温度参数的影响 ━━━━━━━

    温度（temperature）控制生成的多样性：

    temperature = 0.3：非常保守
      → 总是选概率最高的字
      → 生成的文本单调但流畅

    temperature = 0.7：略微保守
      → 大部分时候选高概率的字
      → 偶尔选次高概率的字

    temperature = 1.0：正常
      → 按原始概率分布采样
      → 平衡流畅性和多样性

    temperature = 1.5：比较自由
      → 低概率的字也有机会
      → 更有创造性，但可能不太连贯

    ━━━━━━━ N-Gram vs LSTM ━━━━━━━
    ┌──────────────────────────────────────────────────┐
    │              N-Gram           LSTM                │
    ├──────────────────────────────────────────────────┤
    │  上下文窗口    N-1 个词        理论上无限          │
    │  训练速度      快              慢                  │
    │  生成质量      一般            更好                │
    │  数据需求      中等            大量                │
    │  计算资源      CPU             GPU 推荐            │
    │  可解释性      高              低（黑箱）          │
    └──────────────────────────────────────────────────┘
    """)


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

    lesson_what_is_lm()
    lesson_ngram()
    lesson_text_generation()
    lesson_perplexity()
    lesson_lstm()
    lesson_smoothing()
    lesson_kneser_ney()
    lesson_lstm_training()

    # 课程总结
    print("\n" + "=" * 60)
    print("  第十七章 总结")
    print("=" * 60)
    print("""
    [OK] N-Gram 语言模型 — 基于统计的语言建模
    [OK] 文本生成 — 用语言模型生成新文本
    [OK] 困惑度 — 评估语言模型的指标
    [OK] 平滑技术 — 处理零概率问题
    [OK] Kneser-Ney 平滑 — 续接概率 + 绝对折扣，效果最好
    [OK] LSTM 语言模型 — 真实可训练的神经网络语言模型
    """)

    print("-" * 60)
    print("  下节预告：第十八章 — 深度学习基础")
    print("-" * 60)
    print("""
    下一章我们将学习：
    - 神经网络基础（感知机、激活函数）
    - CNN 和 RNN 的概念
    - LSTM 的详细原理
    - PyTorch 实战

    预习建议：
    - 了解矩阵乘法的基本概念
    - 安装 PyTorch（可选）
    """)
