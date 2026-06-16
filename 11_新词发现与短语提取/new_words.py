"""
==============================================================================
第十一章：新词发现与短语提取
==============================================================================
日期：2026-05-16

同学们好！这节课我们来学习一个非常有趣且实用的无监督 NLP 技术 —— 新词发现与短语提取。

----------------------------------------------------------------------
生活类比：在人群中发现"新面孔"
----------------------------------------------------------------------

想象你是一个社区管理员，每天都有新人搬进来。

你怎么判断一个人是"新邻居"而不是"路人"？

1. 出现频率（互信息）：
   - 如果一个人经常出现在小区里 → 可能是居民
   - 如果一个人只出现一次 → 可能是路人

2. 上下文一致性（左右熵）：
   - 如果这个人的左边总是同一个人 → 关系稳定，是居民
   - 如果这个人的左边总是不同的人 → 关系不稳定，可能是路人

3. 内部凝聚度（互信息）：
   - "张三"总是一起出现 → 是一个整体（一个词）
   - "张 三"经常分开出现 → 不是一个整体

新词发现就是用这三个指标来判断一个字符串是否是一个"词"。

----------------------------------------------------------------------
本章内容
----------------------------------------------------------------------

1. 互信息（Mutual Information）— 衡量内部凝聚度
2. 左右熵（Left/Right Entropy）— 衡量上下文多样性
3. 新词发现算法
4. 短语提取算法

==============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import numpy as np
from collections import Counter, defaultdict


# ==============================================================================
# 第一部分：互信息（Mutual Information）
# ==============================================================================
#
# 互信息衡量两个事件之间的关联程度。
#
# 在新词发现中，我们用互信息来衡量一个字符串的"内部凝聚度"。
#
# 生活类比：
#   想象两个人经常一起出现：
#   - "张三" 和 "李四" 经常一起出现 → 他们可能是朋友
#   - "张三" 和 "今天" 经常一起出现 → 可能只是巧合
#
#   互信息衡量的就是这种"一起出现的程度"是否超过"随机巧合"。
#
# 公式：
#   MI(X, Y) = log2(P(X,Y) / (P(X) * P(Y)))
#
# 其中：
#   P(X,Y) = X 和 Y 一起出现的概率
#   P(X) = X 出现的概率
#   P(Y) = Y 出现的概率
#
# 如果 MI > 0：X 和 Y 正相关（一起出现的频率高于随机）
# 如果 MI = 0：X 和 Y 独立
# 如果 MI < 0：X 和 Y 负相关（一起出现的频率低于随机）
#
# ==============================================================================


def count_ngrams(text: str, n: int) -> dict:
    """
    统计 n-gram 的出现次数

    ━━━━━━━ 什么是 n-gram？ ━━━━━━━
    n-gram 就是连续的 n 个字符。

    例如，对于文本 "我爱北京天安门"：
    - 1-gram（unigram）：我、爱、北、京、天、安、门
    - 2-gram（bigram）：我爱、爱北、北京、京天、天安、安门
    - 3-gram（trigram）：我爱北、爱北京、北京天、京天安、天安门

    参数：
        text: 输入文本
        n: n-gram 的长度

    返回：
        n-gram 计数字典
    """
    ngrams = Counter()
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams[ngram] += 1
    return ngrams


def count_all_ngrams(text: str, max_n: int = 4) -> dict:
    """
    统计所有 n-gram（从 1 到 max_n）

    参数：
        text: 输入文本
        max_n: 最大的 n

    返回：
        所有 n-gram 的计数字典
    """
    all_ngrams = Counter()
    for n in range(1, max_n + 1):
        ngrams = count_ngrams(text, n)
        all_ngrams.update(ngrams)
    return all_ngrams


def compute_mutual_information(word: str, text: str) -> float:
    """
    计算一个词的互信息（内部凝聚度）

    ━━━━━━━ 核心思想 ━━━━━━━
    一个词的互信息，衡量的是这个词"作为整体出现"的程度。

    例如：
    - "北京" 的互信息高 → "北" 和 "京" 总是连在一起出现
    - "的了" 的互信息低 → "的" 和 "了" 经常分开出现

    ━━━━━━━ 公式 ━━━━━━━
    MI(word) = log2(P(word) / (P(word[0]) * P(word[1]) * ... * P(word[-1])))

    其中：
    - P(word) = word 出现的频率
    - P(word[i]) = word 的第 i 个字符出现的频率

    参数：
        word: 要计算互信息的词
        text: 语料库文本

    返回：
        互信息值（越大表示内部凝聚度越高）
    """
    if len(word) < 2:
        return 0.0

    n = len(word)

    # 统计 n-gram
    all_ngrams = count_all_ngrams(text, max_n=n)

    # 总的 n-gram 数量
    total_bigrams = sum(count for gram, count in all_ngrams.items() if len(gram) == 2)
    total_chars = sum(count for gram, count in all_ngrams.items() if len(gram) == 1)

    if total_bigrams == 0 or total_chars == 0:
        return 0.0

    # P(word) = word 出现的次数 / 总的 n-gram 数
    p_word = all_ngrams.get(word, 0) / total_bigrams

    if p_word == 0:
        return 0.0

    # P(word[i]) = 每个字符出现的概率
    p_chars_product = 1.0
    for char in word:
        p_char = all_ngrams.get(char, 0) / total_chars
        if p_char == 0:
            return 0.0
        p_chars_product *= p_char

    # 互信息 = log2(P(word) / (P(char1) * P(char2) * ...))
    mi = math.log2(p_word / p_chars_product) if p_chars_product > 0 else 0.0

    return mi


def compute_pmi(word: str, ngram_counts: dict, total_count: int) -> float:
    """
    计算 Pointwise Mutual Information (PMI)

    ━━━━━━━ PMI vs MI ━━━━━━━
    MI 是期望值（对所有可能的 x, y 求平均）
    PMI 是点值（对特定的 x, y 计算）

    在新词发现中，我们通常用 PMI。

    ━━━━━━━ 注意：分母一致性问题 ━━━━━━━
    原始版本有一个 bug：P(word) 和 P(char) 使用了相同的 total_count。
    但 word 是 n-gram，char 是 unigram，它们的总数不应该相同。

    正确做法：
    - P(word) = count(word) / total_ngram_of_same_length
    - P(char) = count(char) / total_unigram

    例如 word="北京"（bigram）：
    - P("北京") = count("北京") / total_bigrams
    - P("北") = count("北") / total_unigrams
    - P("京") = count("京") / total_unigrams

    参数：
        word: 要计算的词
        ngram_counts: n-gram 计数字典
        total_count: 总的 n-gram 数量（用于兼容旧接口）

    返回：
        PMI 值
    """
    if len(word) < 2:
        return 0.0

    n = len(word)

    # ─── 计算各阶 n-gram 的总数 ───
    # 按长度分组统计，确保分母一致
    totals_by_len = {}
    for gram, count in ngram_counts.items():
        glen = len(gram)
        totals_by_len[glen] = totals_by_len.get(glen, 0) + count

    # P(word)：用同阶 n-gram 的总数作为分母
    total_n = totals_by_len.get(n, 0)
    if total_n == 0:
        return 0.0
    p_word = ngram_counts.get(word, 0) / total_n
    if p_word == 0:
        return 0.0

    # P(每个字符)：用 unigram 的总数作为分母
    total_uni = totals_by_len.get(1, 0)
    if total_uni == 0:
        return 0.0

    p_chars_product = 1.0
    for char in word:
        p_char = ngram_counts.get(char, 0) / total_uni
        if p_char == 0:
            return 0.0
        p_chars_product *= p_char

    return math.log2(p_word / p_chars_product)


# ==============================================================================
# 第二部分：左右熵（Left/Right Entropy）
# ==============================================================================
#
# 左右熵衡量一个词的"上下文多样性"。
#
# 生活类比：
#   想象一个人：
#   - 如果他的左边总是同一个人 → 他的"左熵"低（上下文单一）
#   - 如果他的左边总是不同的人 → 他的"左熵"高（上下文多样）
#
#   左右熵高 → 这个词是一个独立的词（可以和各种词搭配）
#   左右熵低 → 这个词可能不是独立的词（只能和特定词搭配）
#
# 例如：
#   "葡萄" 的左熵高 → 吃葡萄、买葡萄、一串葡萄...
#   "葡" 的左熵低 → 几乎只在 "葡萄" 中出现
#
# ==============================================================================


def compute_left_entropy(word: str, text: str) -> float:
    """
    计算一个词的左熵

    ━━━━━━━ 核心思想 ━━━━━━━
    左熵衡量的是一个词左边字符的多样性。

    例如，对于 "葡萄"：
    - "吃葡萄" → 左边是 "吃"
    - "买葡萄" → 左边是 "买"
    - "一串葡萄" → 左边是 "串"
    → 左边字符多样 → 左熵高

    对于 "葡"：
    - 几乎只在 "葡萄" 中出现 → 左边总是 "萄" 的前一个字
    → 左边字符单一 → 左熵低

    ━━━━━━━ 公式 ━━━━━━━
    左熵 = -Σ P(left_char) * log2(P(left_char))

    其中 P(left_char) 是左边字符出现的频率。

    参数：
        word: 要计算左熵的词
        text: 语料库文本

    返回：
        左熵值（越大表示左边字符越多样）
    """
    # 找到所有 word 出现的位置
    left_chars = []
    for i in range(len(text)):
        if text[i:i+len(word)] == word and i > 0:
            left_chars.append(text[i-1])

    if not left_chars:
        return 0.0

    # 统计左边字符的频率
    char_counts = Counter(left_chars)
    total = len(left_chars)

    # 计算熵
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_right_entropy(word: str, text: str) -> float:
    """
    计算一个词的右熵

    ━━━━━━━ 核心思想 ━━━━━━━
    右熵衡量的是一个词右边字符的多样性。

    例如，对于 "葡萄"：
    - "葡萄干" → 右边是 "干"
    - "葡萄架" → 右边是 "架"
    - "葡萄糖" → 右边是 "糖"
    → 右边字符多样 → 右熵高

    参数：
        word: 要计算右熵的词
        text: 语料库文本

    返回：
        右熵值（越大表示右边字符越多样）
    """
    # 找到所有 word 出现的位置
    right_chars = []
    for i in range(len(text)):
        if text[i:i+len(word)] == word and i + len(word) < len(text):
            right_chars.append(text[i + len(word)])

    if not right_chars:
        return 0.0

    # 统计右边字符的频率
    char_counts = Counter(right_chars)
    total = len(right_chars)

    # 计算熵
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_left_right_entropy(word: str, text: str) -> tuple:
    """
    同时计算左熵和右熵

    参数：
        word: 要计算的词
        text: 语料库文本

    返回：
        (左熵, 右熵)
    """
    return compute_left_entropy(word, text), compute_right_entropy(word, text)


# ==============================================================================
# 第三部分：新词发现算法
# ==============================================================================
#
# 新词发现的核心思路：
#   1. 候选生成：提取所有可能的 n-gram
#   2. 候选过滤：用互信息和左右熵过滤
#   3. 结果排序：按得分排序，输出最可能的新词
#
# 生活类比：
#   想象你在一个派对上找"新人"：
#   1. 先列出所有在场的人（候选生成）
#   2. 过滤掉明显是服务员或路人的（候选过滤）
#   3. 按照"出现次数"和"社交活跃度"排序（结果排序）
#
# ==============================================================================


def extract_candidate_words(text: str, min_len: int = 2, max_len: int = 4,
                             min_freq: int = 3) -> dict:
    """
    提取候选新词（所有频繁出现的 n-gram）

    参数：
        text: 语料库文本
        min_len: 最小词长
        max_len: 最大词长
        min_freq: 最小出现频率

    返回：
        候选词字典 {词: 出现次数}
    """
    candidates = {}

    for n in range(min_len, max_len + 1):
        ngrams = count_ngrams(text, n)
        for ngram, count in ngrams.items():
            if count >= min_freq:
                candidates[ngram] = count

    return candidates


def filter_by_mi(candidates: dict, text: str, mi_threshold: float = 1.0) -> dict:
    """
    使用互信息过滤候选词

    ━━━━━━━ 过滤逻辑 ━━━━━━━
    互信息高 → 内部凝聚度高 → 更可能是一个词
    互信息低 → 内部凝聚度低 → 可能只是巧合拼在一起

    参数：
        candidates: 候选词字典
        text: 语料库文本
        mi_threshold: 互信息阈值

    返回：
        过滤后的候选词字典
    """
    filtered = {}

    for word, count in candidates.items():
        mi = compute_mutual_information(word, text)
        if mi >= mi_threshold:
            filtered[word] = {
                'count': count,
                'mi': mi,
            }

    return filtered


def filter_by_entropy(candidates: dict, text: str,
                      entropy_threshold: float = 0.5) -> dict:
    """
    使用左右熵过滤候选词

    ━━━━━━━ 过滤逻辑 ━━━━━━━
    左右熵高 → 上下文多样 → 更可能是一个独立的词
    左右熵低 → 上下文单一 → 可能只是一个片段

    参数：
        candidates: 候选词字典
        text: 语料库文本
        entropy_threshold: 熵阈值

    返回：
        过滤后的候选词字典
    """
    filtered = {}

    for word, info in candidates.items():
        left_ent, right_ent = compute_left_right_entropy(word, text)

        # 左右熵取较小值（两个方向都要多样）
        min_entropy = min(left_ent, right_ent)

        if min_entropy >= entropy_threshold:
            info['left_entropy'] = left_ent
            info['right_entropy'] = right_ent
            info['min_entropy'] = min_entropy
            filtered[word] = info

    return filtered


def discover_new_words(text: str, min_freq: int = 3, mi_threshold: float = 1.0,
                       entropy_threshold: float = 0.5, min_len: int = 2,
                       max_len: int = 4, top_k: int = 20) -> list:
    """
    新词发现主函数

    ━━━━━━━ 完整流程 ━━━━━━━
    1. 候选生成：提取所有频繁的 n-gram
    2. 互信息过滤：保留内部凝聚度高的
    3. 左右熵过滤：保留上下文多样的
    4. 综合排序：按综合得分排序

    ━━━━━━━ 生活类比 ━━━━━━━
    就像在人群中找"新人"：
    1. 先列出所有出现过的人（候选生成）
    2. 过滤掉只出现一两次的（频率过滤）
    3. 过滤掉总是和同一个人一起出现的（互信息过滤）
    4. 过滤掉上下文太单一的（左右熵过滤）
    5. 按"出现次数"和"社交活跃度"排序（综合排序）

    参数：
        text: 语料库文本
        min_freq: 最小出现频率
        mi_threshold: 互信息阈值
        entropy_threshold: 熵阈值
        min_len: 最小词长
        max_len: 最大词长
        top_k: 返回前 k 个结果

    返回：
        新词列表，每个元素是 (词, 得分信息)
    """
    # 第一步：候选生成
    candidates = extract_candidate_words(text, min_len, max_len, min_freq)

    # 第二步：互信息过滤
    candidates = filter_by_mi(candidates, text, mi_threshold)

    # 第三步：左右熵过滤
    candidates = filter_by_entropy(candidates, text, entropy_threshold)

    # 第四步：综合排序
    # 得分 = 互信息 * min(左熵, 右熵) * log(频率)
    scored_candidates = []
    for word, info in candidates.items():
        score = info['mi'] * info['min_entropy'] * math.log(1 + info['count'])
        scored_candidates.append((word, score, info))

    # 按得分降序排序
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    return scored_candidates[:top_k]


# ==============================================================================
# 第四部分：短语提取算法
# ==============================================================================
#
# 短语提取和新词发现类似，但更关注"多词组合"。
#
# 核心思想：
#   一个短语是由多个词组成的，这些词经常一起出现，
#   而且不能被其他词替换。
#
# 生活类比：
#   想象你在学习成语：
#   - "一举两得" 是一个短语（经常一起出现，不能拆开）
#   - "一 举 两 得" 不是短语（只是四个独立的字）
#
# 常用方法：
#   1. PMI（点互信息）：衡量词与词之间的关联程度
#   2. 左右熵：衡量短语的独立性
#   3. TF-IDF：衡量短语在文档中的重要性
#
# ==============================================================================


def extract_word_pairs(text: str, min_freq: int = 2) -> dict:
    """
    提取词对（两个连续的词）

    参数：
        text: 语料库文本
        min_freq: 最小出现频率

    返回：
        词对字典 {(词1, 词2): 出现次数}
    """
    pairs = Counter()

    # 简单按字符分割
    for i in range(len(text) - 1):
        # 跳过标点
        if text[i] in "，。！？、；：""''（）【】《》 \n":
            continue
        if text[i+1] in "，。！？、；：""''（）【】《》 \n":
            continue
        pair = (text[i], text[i+1])
        pairs[pair] += 1

    # 过滤低频词对
    return {pair: count for pair, count in pairs.items() if count >= min_freq}


def compute_phrase_score(word1: str, word2: str, text: str) -> float:
    """
    计算两个词组成短语的得分

    ━━━━━━━ 得分公式 ━━━━━━━
    score = PMI(word1, word2) * min(left_entropy, right_entropy)

    PMI 高 → 两个词关联紧密
    左右熵高 → 短语是独立的

    参数：
        word1: 第一个词
        word2: 第二个词
        text: 语料库文本

    返回：
        短语得分
    """
    phrase = word1 + word2

    # 计算 PMI
    ngram_counts = count_all_ngrams(text, max_n=2)
    total = sum(ngram_counts.values())

    p_phrase = ngram_counts.get(phrase, 0) / total
    p_word1 = ngram_counts.get(word1, 0) / total
    p_word2 = ngram_counts.get(word2, 0) / total

    if p_phrase == 0 or p_word1 == 0 or p_word2 == 0:
        return 0.0

    pmi = math.log2(p_phrase / (p_word1 * p_word2))

    # 计算左右熵
    left_ent = compute_left_entropy(phrase, text)
    right_ent = compute_right_entropy(phrase, text)
    min_entropy = min(left_ent, right_ent)

    return pmi * max(0, min_entropy)


def extract_phrases(text: str, min_freq: int = 3, top_k: int = 10) -> list:
    """
    提取短语

    ━━━━━━━ 算法步骤 ━━━━━━━
    1. 提取所有候选词对
    2. 计算每个词对的 PMI
    3. 计算每个词对的左右熵
    4. 综合得分并排序

    参数：
        text: 语料库文本
        min_freq: 最小出现频率
        top_k: 返回前 k 个结果

    返回：
        短语列表
    """
    # 提取词对
    pairs = extract_word_pairs(text, min_freq)

    # 计算得分
    scored_phrases = []
    for (w1, w2), count in pairs.items():
        score = compute_phrase_score(w1, w2, text)
        if score > 0:
            scored_phrases.append((w1 + w2, score, count))

    # 排序
    scored_phrases.sort(key=lambda x: x[1], reverse=True)

    return scored_phrases[:top_k]


# ==============================================================================
# 第五部分：辅助工具
# ==============================================================================


def simple_segment(text: str, word_dict: set) -> list:
    """
    使用正向最大匹配进行简单分词

    参数：
        text: 输入文本
        word_dict: 词典

    返回：
        分词结果列表
    """
    result = []
    i = 0
    max_len = max(len(w) for w in word_dict) if word_dict else 1

    while i < len(text):
        matched = False
        # 从最长的可能词开始匹配
        for length in range(min(max_len, len(text) - i), 0, -1):
            word = text[i:i+length]
            if word in word_dict:
                result.append(word)
                i += length
                matched = True
                break

        if not matched:
            result.append(text[i])
            i += 1

    return result


def print_discovery_results(results: list):
    """
    打印新词发现结果

    参数：
        results: 新词发现结果列表
    """
    print(f"\n  {'排名':<4} {'词语':<8} {'得分':>8} {'频率':>6} {'互信息':>8} {'左熵':>6} {'右熵':>6}")
    print("  " + "-" * 55)

    for rank, (word, score, info) in enumerate(results[:15], 1):
        count = info.get('count', 0)
        mi = info.get('mi', 0)
        left_ent = info.get('left_entropy', 0)
        right_ent = info.get('right_entropy', 0)
        print(f"  {rank:<4} {word:<8} {score:>8.2f} {count:>6} {mi:>8.3f} {left_ent:>6.3f} {right_ent:>6.3f}")


def analyze_word(word: str, text: str):
    """
    分析一个词的各项指标

    参数：
        word: 要分析的词
        text: 语料库文本
    """
    print(f"\n  分析词语: '{word}'")
    print(f"  {'='*40}")

    # 频率
    count = text.count(word)
    print(f"  频率: {count}")

    # 互信息
    mi = compute_mutual_information(word, text)
    print(f"  互信息: {mi:.4f}")

    # 左右熵
    left_ent, right_ent = compute_left_right_entropy(word, text)
    print(f"  左熵: {left_ent:.4f}")
    print(f"  右熵: {right_ent:.4f}")
    print(f"  最小熵: {min(left_ent, right_ent):.4f}")

    # 综合得分
    score = mi * min(left_ent, right_ent) * math.log(1 + count)
    print(f"  综合得分: {score:.4f}")


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十一章                      ║
    ║        新词发现与短语提取                              ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 测试文本
    test_text = """
    今天天气真好，我去北京大学看望小明。
    小明在北京大学学习自然语言处理。
    自然语言处理是人工智能的重要方向。
    人工智能技术发展迅速，深度学习是核心技术。
    深度学习在自然语言处理中应用广泛。
    北京大学的自然语言处理实验室很厉害。
    """

    print("=" * 60)
    print("测试：互信息计算")
    print("=" * 60)

    test_words = ["北京", "大学", "自然", "语言", "处理", "人工", "智能"]
    for word in test_words:
        mi = compute_mutual_information(word, test_text)
        print(f"  MI('{word}') = {mi:.4f}")

    print("\n" + "=" * 60)
    print("测试：左右熵计算")
    print("=" * 60)

    for word in test_words:
        left_ent, right_ent = compute_left_right_entropy(word, test_text)
        print(f"  '{word}': 左熵={left_ent:.4f}, 右熵={right_ent:.4f}")

    print("\n" + "=" * 60)
    print("测试：新词发现")
    print("=" * 60)

    results = discover_new_words(test_text, min_freq=2, mi_threshold=0.5,
                                  entropy_threshold=0.3, top_k=10)
    print_discovery_results(results)


# ==============================================================================
# 第六部分：迭代式新词发现
# ==============================================================================
#
# 前面的算法只做一轮发现，但实际上：
# 第一轮发现的新词可以帮助我们更好地分割文本，
# 从而在第二轮发现更多新词。
#
# 生活类比：
#   想象你在拼图：
#   - 第一轮：先把最容易的边角拼好（高频、高凝聚度的词）
#   - 第二轮：拼好边角后，中间的部分更容易辨认了
#   - 第三轮：越来越完整，越来越容易
#
#   每一轮的发现都会帮助下一轮发现更多。
#   直到没有新发现为止（收敛）。
#
# ==============================================================================


class IterativeWordDiscovery:
    """
    迭代式新词发现器

    ━━━━━━━ 核心思想 ━━━━━━━
    1. 第一轮：用传统方法发现新词
    2. 用发现的新词对文本进行分词
    3. 在分词后的文本上重新发现新词
    4. 重复直到没有新词被发现（收敛）

    ━━━━━━━ 为什么迭代有效？ ━━━━━━━
    想象文本 "北京大学自然语言处理实验室"

    第一轮发现："北京"、"大学"、"自然"、"语言"、"处理"
    用这些词分词：["北京", "大学", "自然", "语言", "处理", "实验室"]

    第二轮发现："实验室"、"自然语言处理"（因为分词后这些词的频率提高了）
    用新词更新分词：["北京", "大学", "自然语言处理", "实验室"]

    第三轮发现："北京大学"、"自然语言处理实验室"
    最终分词：["北京大学", "自然语言处理实验室"]

    每一轮都在上一轮的基础上发现更长、更完整的新词！

    ━━━━━━━ 收敛判断 ━━━━━━━
    当某一轮没有发现任何新词时，说明已经收敛了。
    就像拼图拼完了，再拼也没有新的进展。

    属性：
        min_freq: 最小出现频率
        mi_threshold: 互信息阈值
        entropy_threshold: 左右熵阈值
        min_len: 最小词长
        max_len: 最大词长
    """

    def __init__(self, min_freq: int = 2, mi_threshold: float = 1.0,
                 entropy_threshold: float = 0.5, min_len: int = 2,
                 max_len: int = 4):
        """
        初始化迭代式新词发现器

        ━━━━━━━ 生活类比 ━━━━━━━
        就像设置拼图的规则：
        - 最少出现几次才算"常见图案"（min_freq）
        - 图案内部要多"凝聚"才算完整（mi_threshold）
        - 图案周围要多"多样"才算独立（entropy_threshold）
        - 最小/最大的拼图块尺寸（min_len, max_len）

        参数：
            min_freq: 最小出现频率
            mi_threshold: 互信息阈值
            entropy_threshold: 左右熵阈值
            min_len: 最小词长
            max_len: 最大词长
        """
        self.min_freq = min_freq
        self.mi_threshold = mi_threshold
        self.entropy_threshold = entropy_threshold
        self.min_len = min_len
        self.max_len = max_len

    def discover(self, text: str, iterations: int = 3) -> dict:
        """
        运行迭代式新词发现

        ━━━━━━━ 算法流程 ━━━━━━━
        for 每次迭代:
            1. 在当前文本上发现新词
            2. 用已发现的所有词对文本进行分词
            3. 用分词后的文本替换原始文本
            4. 检查是否收敛（没有新词被发现）

        ━━━━━━━ 生活类比 ━━━━━━━
        就像考古学家清理化石：
        - 第一遍：用大刷子扫掉表面的土（发现大块的词）
        - 第二遍：用小刷子清理细节（发现更精细的词）
        - 第三遍：用放大镜观察（发现最细微的结构）
        - 每一遍都比上一遍更精细

        参数：
            text: 原始文本
            iterations: 最大迭代次数

        返回：
            {
                'all_words': 所有发现的词集合,
                'iterations': 每轮迭代的详情列表,
                'converged': 是否收敛,
                'final_iteration': 最终迭代次数
            }
        """
        # 所有已发现的词
        all_words = set()
        # 每轮迭代的记录
        iteration_details = []
        # 当前文本（会随着迭代变化）
        current_text = text

        print("  开始迭代式新词发现...")
        print(f"  最大迭代次数: {iterations}")
        print(f"  参数: min_freq={self.min_freq}, mi={self.mi_threshold}, "
              f"entropy={self.entropy_threshold}")
        print()

        for it in range(iterations):
            print(f"  ─── 第 {it + 1} 轮迭代 ───")

            # 第一步：在当前文本上发现新词
            results = discover_new_words(
                current_text,
                min_freq=self.min_freq,
                mi_threshold=self.mi_threshold,
                entropy_threshold=self.entropy_threshold,
                min_len=self.min_len,
                max_len=self.max_len,
                top_k=50,
            )

            # 提取本轮发现的新词
            new_words = set()
            for word, score, info in results:
                if word not in all_words:
                    new_words.add(word)

            # 记录本轮详情
            detail = {
                'iteration': it + 1,
                'new_words_count': len(new_words),
                'new_words': new_words,
                'total_words': len(all_words) + len(new_words),
                'text_length': len(current_text),
            }
            iteration_details.append(detail)

            print(f"    发现新词 {len(new_words)} 个: "
                  f"{', '.join(sorted(new_words)[:10])}"
                  f"{'...' if len(new_words) > 10 else ''}")

            # 第二步：检查是否收敛
            if not new_words:
                print(f"    收敛！没有发现新词。")
                break

            # 第三步：更新已发现词集合
            all_words.update(new_words)

            # 第四步：用所有已发现的词对文本进行分词
            # 分词后的文本中，已发现的词会作为整体出现
            # 这样在下一轮中，这些词的频率会更准确
            word_dict = all_words | set(text)  # 词典 = 已发现的词 + 所有单字
            segments = simple_segment(current_text, word_dict)

            # 用空格连接分词结果，这样已发现的词在下一轮中
            # 会作为整体被统计，而不是被拆成单个字符
            current_text = " ".join(segments)

            print(f"    当前词典大小: {len(all_words)}")
            print(f"    分词后文本长度: {len(current_text)}")
            print()

        converged = (len(iteration_details) > 0 and
                     iteration_details[-1]['new_words_count'] == 0)

        print(f"  迭代结束！共发现 {len(all_words)} 个新词。")
        print(f"  是否收敛: {'是' if converged else '否'}")

        return {
            'all_words': all_words,
            'iterations': iteration_details,
            'converged': converged,
            'final_iteration': len(iteration_details),
        }


def demo_iterative_discovery():
    """
    演示迭代式新词发现

    ━━━━━━━ 演示内容 ━━━━━━━
    1. 使用较长的文本进行迭代发现
    2. 观察每轮发现的新词数量变化
    3. 验证收敛性
    """
    print("=" * 60)
    print("迭代式新词发现")
    print("=" * 60)

    print("""
    ━━━━━━━ 迭代式发现 vs 一次性发现 ━━━━━━━

    一次性发现：
      只做一轮，发现高频、高凝聚度的词
      可能漏掉一些需要"上下文"才能识别的词

    迭代式发现：
      第一轮发现的基础词 → 帮助第二轮发现更长的词
      第二轮发现的长词 → 帮助第三轮发现更完整的短语
      直到没有新发现为止

    就像考古挖掘：
      第一层：挖出大件文物
      第二层：大件清理后，露出更多小件
      第三层：小件清理后，发现最精细的细节
    """)

    # 使用较长的测试文本
    test_text = """
    自然语言处理是人工智能的重要研究方向。北京大学自然语言处理实验室是国内顶尖的
    研究机构。深度学习技术在自然语言处理中应用广泛，机器学习算法是核心技术。
    机器学习和深度学习密切相关，都是人工智能的核心技术。
    自然语言处理需要机器学习的支持，也需要大规模语料库。
    北京大学清华大学都是中国顶尖的高等学府，培养了大量人工智能人才。
    人工智能技术发展迅速，深度学习自然语言处理计算机视觉都是热门方向。
    计算机视觉和自然语言处理是人工智能的两大核心方向。
    深度学习模型在自然语言处理任务中表现优异。
    预训练语言模型是自然语言处理的最新突破。
    大规模预训练模型改变了自然语言处理的研究范式。
    """

    print("  测试文本长度:", len(test_text), "字符")
    print()

    # 运行迭代式发现
    discovery = IterativeWordDiscovery(
        min_freq=2,
        mi_threshold=0.5,
        entropy_threshold=0.3,
        min_len=2,
        max_len=4,
    )
    result = discovery.discover(test_text, iterations=5)

    # 打印每轮迭代的统计
    print("\n  迭代统计:")
    print(f"  {'轮次':<6} {'新词数':<8} {'累计词数':<10} {'文本长度':<10}")
    print("  " + "-" * 35)
    for detail in result['iterations']:
        print(f"  {detail['iteration']:<6} "
              f"{detail['new_words_count']:<8} "
              f"{detail['total_words']:<10} "
              f"{detail['text_length']:<10}")

    # 打印所有发现的词
    print(f"\n  所有发现的词 ({len(result['all_words'])} 个):")
    sorted_words = sorted(result['all_words'], key=len, reverse=True)
    for i, word in enumerate(sorted_words):
        if i > 0 and i % 10 == 0:
            print()
        print(f"    {word:<10}", end="")
    print()

    print(f"""
    ━━━━━━━ 分析 ━━━━━━━
    1. 第一轮发现的通常是高频的短词（2-3 字）
    2. 后续轮次发现更长的词（因为短词被合并后，长词的频率提高了）
    3. 收敛后没有新词被发现，说明算法达到了稳定状态

    迭代式方法的优点：
    - 能发现更长、更完整的新词
    - 减少了"碎片化"问题
    - 结果更符合人类的词语认知

    迭代式方法的局限：
    - 计算量比一次性发现大
    - 可能在某些情况下"过度合并"
    - 需要合理设置迭代次数
""")


# =============================================
# 课程总结
# =============================================
"""
核心收获：
- 互信息（PMI）衡量字符串的内部凝聚度，PMI 越高说明字符组合越不可能是随机拼凑
- 左右熵衡量上下文多样性，熵越高说明该词可以和越多不同的词搭配，是独立词的概率越大
- 迭代式新词发现通过多轮递进式挖掘，先找短词再合并成长词，逐步逼近完整词典

常见陷阱：
- PMI 计算时分母不一致（用 unigram 总数除 bigram 概率），导致数值偏差
- 只用互信息过滤而忽略左右熵，会把"的了""是的"等非词组合误判为新词
- 频率阈值设得太低会引入大量噪声，设得太高会漏掉低频但有意义的新词

下节课预告：
- 我们将学习搜索引擎的基本原理，把前面学过的分词、关键词提取、相似度计算串联起来
"""
