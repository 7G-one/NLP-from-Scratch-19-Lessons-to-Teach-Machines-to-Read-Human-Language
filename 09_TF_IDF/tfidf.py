"""
==============================================================================
第九章：TF-IDF — 词频-逆文档频率
==============================================================================
日期：2026-05-16

同学们好！这节课我们来学习一个非常经典且实用的文本特征提取方法 —— TF-IDF。

----------------------------------------------------------------------
生活类比：在一摞报纸中找关键词
----------------------------------------------------------------------

想象你的书桌上堆了 100 份报纸，你想知道每份报纸在"讲什么"。

方法：
  1. 数一数每个词在某份报纸中出现了多少次 → 这就是 TF（词频）
     比如"股市"在经济版面出现了 15 次 → TF 很高

  2. 但是"的"、"是"、"了"这些词在每份报纸中都出现了很多次！
     虽然它们的 TF 很高，但它们对区分"报纸主题"毫无帮助。
     所以我们要降低这些"到处都出现"的词的权重。
     → 这就是 IDF（逆文档频率）

  3. TF * IDF = 一个词在这份报纸中的"重要程度"
     - "股市"：TF 高（出现多次） × IDF 高（只在经济版面出现）→ 重要！
     - "的"：  TF 高（出现多次） × IDF 低（每份报纸都有） → 不重要！

TF-IDF 的核心思想：
  一个词的重要性，不仅取决于它在当前文档中出现的频率，
  还取决于它在整个文档集合中的"稀有程度"。

----------------------------------------------------------------------
本章内容
----------------------------------------------------------------------

1. TF（词频）计算
2. IDF（逆文档频率）计算
3. TF-IDF 计算（从零实现）
4. sklearn TfidfVectorizer 的使用
5. 关键词提取
6. 文档相似度

==============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math
import numpy as np


# ==============================================================================
# 第一部分：TF（词频，Term Frequency）
# ==============================================================================
#
# TF 衡量一个词在一篇文档中出现的频率。
#
# 公式：
#   TF(t, d) = 词 t 在文档 d 中出现的次数 / 文档 d 的总词数
#
# 生活类比：
#   想象你在数一篇作文中每个字出现了几次。
#   如果作文有 100 个字，"爱" 出现了 10 次，
#   那么 TF("爱", 作文) = 10/100 = 0.1
#
# 注意：
#   TF 只看"当前文档"，不看其他文档。
#   所以"的"的 TF 可能很高，但它是停用词，没有实际意义。
#
# ==============================================================================


def compute_tf_raw(term: str, document: list) -> int:
    """
    计算词在文档中的原始词频（出现次数）

    参数：
        term: 要查询的词
        document: 文档的分词结果列表

    返回：
        词出现的次数

    示例：
        >>> compute_tf_raw("苹果", ["我", "喜欢", "吃", "苹果", "苹果"])
        2
    """
    return document.count(term)


def compute_tf_ratio(term: str, document: list) -> float:
    """
    计算词在文档中的词频比率

    ━━━━━━━ 公式 ━━━━━━━
    TF(t, d) = count(t in d) / len(d)

    参数：
        term: 要查询的词
        document: 文档的分词结果列表

    返回：
        词频比率（0 到 1 之间）

    示例：
        >>> compute_tf_ratio("苹果", ["我", "喜欢", "吃", "苹果", "苹果"])
        0.4
    """
    if not document:
        return 0.0
    return document.count(term) / len(document)


def compute_tf_log(term: str, document: list) -> float:
    """
    计算对数词频（Log Normalization）

    ━━━━━━━ 公式 ━━━━━━━
    TF(t, d) = log(1 + count(t in d))

    为什么要用对数？
    直接用词频的话，出现 100 次的词比出现 10 次的词重要 10 倍。
    但实际上，出现 100 次和出现 10 次的差别没那么大。
    对数可以"压缩"这种差距。

    生活类比：
      你有 1 元钱和 100 元钱，差别很大。
      但你有 1 万元和 100 万元，差别就"感觉上"小多了。
      对数做的就是这种"压缩"。

    参数：
        term: 要查询的词
        document: 文档的分词结果列表

    返回：
        对数词频
    """
    raw_count = document.count(term)
    return math.log(1 + raw_count)


def compute_tf_augmented(term: str, document: list) -> float:
    """
    计算增强词频（Augmented TF）

    ━━━━━━━ 公式 ━━━━━━━
    TF(t, d) = 0.5 + 0.5 * count(t in d) / max_count

    其中 max_count 是文档中出现次数最多的词的频率。

    为什么要增强？
    防止长文档的词频天然偏高。
    一篇 10000 字的文章，"的" 出现 500 次；
    一篇 100 字的文章，"的" 出现 5 次。
    但两篇文章中"的"的重要性其实差不多！

    参数：
        term: 要查询的词
        document: 文档的分词结果列表

    返回：
        增强词频
    """
    if not document:
        return 0.0

    # 统计每个词的出现次数
    word_counts = {}
    for word in document:
        word_counts[word] = word_counts.get(word, 0) + 1

    # 找到出现次数最多的词的频率
    max_count = max(word_counts.values()) if word_counts else 1

    # 计算增强词频
    count = document.count(term)
    return 0.5 + 0.5 * count / max_count


def compute_tf_document(document: list, method: str = "ratio") -> dict:
    """
    计算整篇文档的 TF 向量

    参数：
        document: 文档的分词结果列表
        method: 计算方法，可选 "ratio", "log", "augmented"

    返回：
        词频字典 {词: TF值}
    """
    tf_vector = {}
    unique_terms = set(document)

    for term in unique_terms:
        if method == "ratio":
            tf_vector[term] = compute_tf_ratio(term, document)
        elif method == "log":
            tf_vector[term] = compute_tf_log(term, document)
        elif method == "augmented":
            tf_vector[term] = compute_tf_augmented(term, document)
        else:
            tf_vector[term] = compute_tf_ratio(term, document)

    return tf_vector


# ==============================================================================
# 第二部分：IDF（逆文档频率，Inverse Document Frequency）
# ==============================================================================
#
# IDF 衡量一个词在所有文档中的"稀有程度"。
#
# 公式：
#   IDF(t) = log(总文档数 / 包含词 t 的文档数)
#
# 生活类比：
#   想象你有 100 份报纸：
#   - "股市" 只在 5 份经济报纸中出现 → IDF 高 → 这个词很"稀有"，很有区分度
#   - "的" 在所有 100 份报纸中都出现 → IDF 低 → 这个词很"常见"，没区分度
#
# IDF 的作用：
#   降低那些"到处都出现"的词的权重，
#   提高那些"只在少数文档中出现"的词的权重。
#
# ==============================================================================


def compute_idf_raw(term: str, documents: list) -> float:
    """
    计算原始 IDF

    ━━━━━━━ 公式 ━━━━━━━
    IDF(t) = log(N / df)

    其中：
    - N = 总文档数
    - df = 包含词 t 的文档数

    参数：
        term: 要查询的词
        documents: 文档集合（每个文档是分词结果列表）

    返回：
        IDF 值
    """
    n = len(documents)

    # 计算包含该词的文档数
    df = sum(1 for doc in documents if term in doc)

    # 避免除以零（如果没有任何文档包含该词）
    if df == 0:
        return 0.0

    return math.log(n / df)


def compute_idf_smooth(term: str, documents: list) -> float:
    """
    计算平滑 IDF

    ━━━━━━━ 公式 ━━━━━━━
    IDF(t) = log((1 + N) / (1 + df)) + 1

    为什么要平滑？
    如果某个词不在任何文档中，原始 IDF 会出现 log(无穷大) 的问题。
    平滑版本可以避免这种情况。

    生活类比：
      就像考试评分：如果某个题目所有人都没做对，
      我们不会给这道题"无穷大的难度"，
      而是给一个"非常大的有限值"。

    参数：
        term: 要查询的词
        documents: 文档集合

    返回：
        平滑 IDF 值
    """
    n = len(documents)
    df = sum(1 for doc in documents if term in doc)
    return math.log((1 + n) / (1 + df)) + 1


def compute_idf_probabilistic(term: str, documents: list) -> float:
    """
    计算概率 IDF

    ━━━━━━━ 公式 ━━━━━━━
    IDF(t) = log((N - df) / df)

    这个版本的 IDF 直接反映了"词不出现的概率 vs 出现的概率"的比值。

    参数：
        term: 要查询的词
        documents: 文档集合

    返回：
        概率 IDF 值
    """
    n = len(documents)
    df = sum(1 for doc in documents if term in doc)

    if df == 0 or df == n:
        return 0.0

    return math.log((n - df) / df)


def compute_idf_corpus(documents: list, method: str = "smooth") -> dict:
    """
    计算整个语料库的 IDF 向量

    参数：
        documents: 文档集合
        method: 计算方法，可选 "raw", "smooth", "probabilistic"

    返回：
        IDF 字典 {词: IDF值}
    """
    # 收集所有不重复的词
    all_terms = set()
    for doc in documents:
        all_terms.update(doc)

    idf_vector = {}
    for term in all_terms:
        if method == "raw":
            idf_vector[term] = compute_idf_raw(term, documents)
        elif method == "smooth":
            idf_vector[term] = compute_idf_smooth(term, documents)
        elif method == "probabilistic":
            idf_vector[term] = compute_idf_probabilistic(term, documents)
        else:
            idf_vector[term] = compute_idf_smooth(term, documents)

    return idf_vector


# ==============================================================================
# 第三部分：TF-IDF 计算
# ==============================================================================
#
# TF-IDF = TF * IDF
#
# 生活类比：
#   回到"一摞报纸"的例子：
#
#   TF：一篇报纸中"股市"出现了 15 次 → TF 高 → 在这篇报纸中很重要
#   IDF："股市"只在 5% 的报纸中出现 → IDF 高 → 这个词很有区分度
#   TF-IDF = 15 * 很高 = 非常高 → "股市"是这篇报纸的关键词！
#
#   反过来：
#   TF：一篇报纸中"的"出现了 200 次 → TF 很高
#   IDF："的"在 100% 的报纸中都出现 → IDF 很低（接近 0）
#   TF-IDF = 200 * 接近0 = 接近0 → "的"不是关键词！
#
# ==============================================================================


def compute_tfidf(tf_vector: dict, idf_vector: dict) -> dict:
    """
    计算 TF-IDF 向量

    ━━━━━━━ 公式 ━━━━━━━
    TF-IDF(t, d) = TF(t, d) * IDF(t)

    参数：
        tf_vector: 词频向量 {词: TF值}
        idf_vector: 逆文档频率向量 {词: IDF值}

    返回：
        TF-IDF 向量 {词: TF-IDF值}
    """
    tfidf_vector = {}

    for term, tf_value in tf_vector.items():
        idf_value = idf_vector.get(term, 0.0)
        tfidf_vector[term] = tf_value * idf_value

    return tfidf_vector


def compute_tfidf_corpus(documents: list, tf_method: str = "ratio",
                          idf_method: str = "smooth") -> list:
    """
    计算整个语料库的 TF-IDF 矩阵

    ━━━━━━━ 步骤 ━━━━━━━
    1. 对每篇文档计算 TF
    2. 对整个语料库计算 IDF
    3. TF * IDF = TF-IDF

    参数：
        documents: 文档集合
        tf_method: TF 计算方法
        idf_method: IDF 计算方法

    返回：
        TF-IDF 矩阵（每篇文档一个字典）
    """
    # 第一步：计算整个语料库的 IDF
    idf_vector = compute_idf_corpus(documents, method=idf_method)

    # 第二步：对每篇文档计算 TF-IDF
    tfidf_matrix = []
    for doc in documents:
        tf_vector = compute_tf_document(doc, method=tf_method)
        tfidf_vector = compute_tfidf(tf_vector, idf_vector)
        tfidf_matrix.append(tfidf_vector)

    return tfidf_matrix


def tfidf_to_numpy_matrix(documents: list) -> tuple:
    """
    将 TF-IDF 矩阵转换为 numpy 数组

    参数：
        documents: 文档集合

    返回：
        (矩阵, 词汇表)
        矩阵形状为 (文档数, 词汇表大小)
    """
    # 计算 TF-IDF
    tfidf_matrix = compute_tfidf_corpus(documents)

    # 构建词汇表
    vocab = sorted(set(term for doc in documents for term in doc))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # 转换为 numpy 矩阵
    n_docs = len(documents)
    n_terms = len(vocab)
    matrix = np.zeros((n_docs, n_terms))

    for doc_idx, tfidf_vector in enumerate(tfidf_matrix):
        for term, value in tfidf_vector.items():
            col_idx = word_to_idx[term]
            matrix[doc_idx][col_idx] = value

    return matrix, vocab


# ==============================================================================
# 第四部分：sklearn TfidfVectorizer 使用
# ==============================================================================
#
# 在实际项目中，我们通常使用 sklearn 的 TfidfVectorizer，
# 它已经帮我们封装好了所有细节。
#
# 生活类比：
#   从零实现 TF-IDF 就像自己做一道菜：买菜、洗菜、切菜、炒菜...
#   用 sklearn 就像点外卖：一行代码搞定！
#
#   但学习自己做菜（从零实现）能让我们理解每一步的原理，
#   这样在用"外卖"（sklearn）时才能更好地调参和 debug。
#
# ==============================================================================


def sklearn_tfidf_demo(documents: list) -> tuple:
    """
    使用 sklearn 的 TfidfVectorizer 计算 TF-IDF

    参数：
        documents: 文档列表（字符串列表，每篇文档是一个字符串）

    返回：
        (特征矩阵, 特征名列表)
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("  [提示] sklearn 未安装，请运行: pip install scikit-learn")
        return None, None

    # 创建 TfidfVectorizer 对象
    # analyzer='char' 表示按字符分词（适用于中文）
    # 如果是英文，可以用默认的 analyzer='word'
    vectorizer = TfidfVectorizer(
        analyzer='char',        # 按字符分词
        max_features=1000,      # 最多保留 1000 个特征
        min_df=1,               # 词至少出现在 1 篇文档中
    )

    # 计算 TF-IDF 矩阵
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 获取特征名（词汇表）
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names


def sklearn_tfidf_keywords(documents: list, top_k: int = 5) -> list:
    """
    使用 sklearn 提取每篇文档的关键词

    参数：
        documents: 文档列表
        top_k: 每篇文档提取的关键词数量

    返回：
        每篇文档的关键词列表
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("  [提示] sklearn 未安装")
        return []

    # 使用 jieba 分词（如果可用）
    def chinese_tokenizer(text):
        try:
            import jieba
            return [w for w in jieba.cut(text) if len(w.strip()) > 0]
        except ImportError:
            return list(text)

    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        max_features=1000,
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    all_keywords = []
    for doc_idx in range(len(documents)):
        # 获取该文档的 TF-IDF 值
        tfidf_scores = tfidf_matrix[doc_idx].toarray().flatten()

        # 按 TF-IDF 值排序
        top_indices = tfidf_scores.argsort()[-top_k:][::-1]

        # 提取关键词
        keywords = []
        for idx in top_indices:
            if tfidf_scores[idx] > 0:
                keywords.append((feature_names[idx], tfidf_scores[idx]))
        all_keywords.append(keywords)

    return all_keywords


# ==============================================================================
# 第五部分：关键词提取
# ==============================================================================
#
# TF-IDF 最常见的应用之一就是关键词提取。
#
# 核心思想：
#   对于一篇文档，计算每个词的 TF-IDF 值，
#   TF-IDF 值最高的词就是这篇文档的关键词。
#
# 生活类比：
#   就像你在做读书笔记时，会用荧光笔标出"重点词"。
#   TF-IDF 就是自动帮你"画荧光笔"的工具。
#
# ==============================================================================


def extract_keywords_tfidf(document: list, documents: list,
                            top_k: int = 5) -> list:
    """
    使用 TF-IDF 提取一篇文档的关键词

    ━━━━━━━ 步骤 ━━━━━━━
    1. 计算文档中每个词的 TF
    2. 计算语料库中每个词的 IDF
    3. TF * IDF = TF-IDF
    4. 返回 TF-IDF 最高的 top_k 个词

    参数：
        document: 目标文档的分词结果
        documents: 整个语料库（包含目标文档）
        top_k: 返回的关键词数量

    返回：
        关键词列表，每个元素是 (词, TF-IDF值)
    """
    # 计算 TF
    tf_vector = compute_tf_document(document, method="ratio")

    # 计算 IDF
    idf_vector = compute_idf_corpus(documents, method="smooth")

    # 计算 TF-IDF
    tfidf_vector = compute_tfidf(tf_vector, idf_vector)

    # 按 TF-IDF 值降序排序
    sorted_terms = sorted(tfidf_vector.items(), key=lambda x: x[1], reverse=True)

    # 返回前 top_k 个
    return sorted_terms[:top_k]


def extract_keywords_all_documents(documents: list, top_k: int = 5) -> list:
    """
    批量提取每篇文档的关键词

    参数：
        documents: 文档集合
        top_k: 每篇文档提取的关键词数量

    返回：
        每篇文档的关键词列表
    """
    all_keywords = []
    for doc in documents:
        keywords = extract_keywords_tfidf(doc, documents, top_k)
        all_keywords.append(keywords)
    return all_keywords


# ==============================================================================
# 第六部分：文档相似度
# ==============================================================================
#
# TF-IDF 另一个重要应用是计算文档相似度。
#
# 核心思想：
#   1. 把每篇文档用 TF-IDF 向量表示
#   2. 计算两个向量的余弦相似度
#   余弦相似度越高 → 文档越相似
#
# 生活类比：
#   想象你有两篇文章，把它们各自变成一组数字（TF-IDF 向量）。
#   然后比较这两组数字的"方向"是否一致。
#   方向越一致 → 文章越相似。
#
# ==============================================================================


def cosine_similarity_numpy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    使用 numpy 计算两个向量的余弦相似度

    参数：
        vec1: 第一个向量
        vec2: 第二个向量

    返回：
        余弦相似度
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def document_similarity(doc1: list, doc2: list, documents: list) -> float:
    """
    计算两篇文档的 TF-IDF 余弦相似度

    ━━━━━━━ 步骤 ━━━━━━━
    1. 计算所有文档的 TF-IDF 矩阵
    2. 取出两篇文档的 TF-IDF 向量
    3. 计算余弦相似度

    参数：
        doc1: 第一篇文档的分词结果
        doc2: 第二篇文档的分词结果
        documents: 整个语料库

    返回：
        文档相似度（0 到 1）
    """
    # 计算 TF-IDF 矩阵
    matrix, vocab = tfidf_to_numpy_matrix(documents)

    # 找到 doc1 和 doc2 在语料库中的索引
    idx1 = None
    idx2 = None
    for i, doc in enumerate(documents):
        if doc == doc1 and idx1 is None:
            idx1 = i
        elif doc == doc2 and idx2 is None:
            idx2 = i

    if idx1 is None or idx2 is None:
        return 0.0

    # 计算余弦相似度
    return cosine_similarity_numpy(matrix[idx1], matrix[idx2])


def document_similarity_pair(doc1: list, doc2: list) -> float:
    """
    仅基于两篇文档计算相似度（不依赖其他文档）

    ━━━━━━━ 适用场景 ━━━━━━━
    当你只有两篇文档，没有整个语料库时使用。
    这种情况下 IDF 无法计算，所以直接用 TF 向量的余弦相似度。

    参数：
        doc1: 第一篇文档的分词结果
        doc2: 第二篇文档的分词结果

    返回：
        文档相似度
    """
    # 构建词汇表
    vocab = sorted(set(doc1 + doc2))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # 构建 TF 向量
    vec1 = np.zeros(len(vocab))
    vec2 = np.zeros(len(vocab))

    for word in doc1:
        vec1[word_to_idx[word]] += 1
    for word in doc2:
        vec2[word_to_idx[word]] += 1

    # 归一化
    if np.sum(vec1) > 0:
        vec1 = vec1 / np.sum(vec1)
    if np.sum(vec2) > 0:
        vec2 = vec2 / np.sum(vec2)

    return cosine_similarity_numpy(vec1, vec2)


# ==============================================================================
# 第七部分：辅助工具
# ==============================================================================


def simple_tokenize(text: str) -> list:
    """
    简单的中文分词（按字符分词）

    在实际项目中，应该使用 jieba 等专业分词工具。

    参数：
        text: 输入文本

    返回：
        分词结果列表
    """
    tokens = []
    for char in text:
        if char.strip() and char not in "，。！？、；：""''（）【】《》 ":
            tokens.append(char)
    return tokens


def jieba_tokenize(text: str) -> list:
    """
    使用 jieba 进行分词

    如果 jieba 未安装，回退到简单分词。
    """
    try:
        import jieba
        return [w for w in jieba.cut(text) if len(w.strip()) > 0]
    except ImportError:
        return simple_tokenize(text)


def print_tfidf_table(tfidf_vector: dict, top_k: int = 10):
    """
    打印 TF-IDF 表格

    参数：
        tfidf_vector: TF-IDF 字典
        top_k: 显示前 k 个
    """
    sorted_terms = sorted(tfidf_vector.items(), key=lambda x: x[1], reverse=True)

    print(f"  {'词':<8} {'TF-IDF':>10}  {'可视化'}")
    print("  " + "-" * 40)

    for term, score in sorted_terms[:top_k]:
        bar = "#" * int(score * 50)
        print(f"  {term:<8} {score:>10.4f}  {bar}")


def compare_tfidf_methods(document: list, documents: list):
    """
    对比不同的 TF-IDF 计算方法

    参数：
        document: 目标文档
        documents: 语料库
    """
    methods = [
        ("TF=比例 + IDF=平滑", "ratio", "smooth"),
        ("TF=对数 + IDF=平滑", "log", "smooth"),
        ("TF=增强 + IDF=平滑", "augmented", "smooth"),
        ("TF=比例 + IDF=原始", "ratio", "raw"),
    ]

    print(f"\n  {'方法':<20} {'Top-3 关键词'}")
    print("  " + "-" * 50)

    for name, tf_method, idf_method in methods:
        tf_vec = compute_tf_document(document, method=tf_method)
        idf_vec = compute_idf_corpus(documents, method=idf_method)
        tfidf_vec = compute_tfidf(tf_vec, idf_vec)

        sorted_terms = sorted(tfidf_vec.items(), key=lambda x: x[1], reverse=True)
        top3 = [f"{t}({s:.3f})" for t, s in sorted_terms[:3]]
        print(f"  {name:<20} {', '.join(top3)}")


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第九章                        ║
    ║        TF-IDF                                       ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 测试数据：模拟一篇文档的分词结果
    test_doc = ["我", "喜欢", "吃", "苹果", "苹果", "很", "好吃"]
    print("=" * 60)
    print("测试：TF 计算")
    print("=" * 60)
    print(f"  文档: {test_doc}")
    print(f"  TF('苹果', 文档) = {compute_tf_ratio('苹果', test_doc):.4f}")
    print(f"  TF('喜欢', 文档) = {compute_tf_ratio('喜欢', test_doc):.4f}")

    # 测试数据：模拟语料库
    corpus = [
        ["我", "喜欢", "吃", "苹果"],
        ["我", "喜欢", "运动"],
        ["今天", "天气", "很", "好"],
        ["苹果", "手机", "很", "好用"],
    ]

    print("\n" + "=" * 60)
    print("测试：IDF 计算")
    print("=" * 60)
    print(f"  语料库大小: {len(corpus)} 篇文档")
    print(f"  IDF('我') = {compute_idf_smooth('我', corpus):.4f}")
    print(f"  IDF('苹果') = {compute_idf_smooth('苹果', corpus):.4f}")
    print(f"  IDF('运动') = {compute_idf_smooth('运动', corpus):.4f}")

    print("\n" + "=" * 60)
    print("测试：TF-IDF 关键词提取")
    print("=" * 60)
    keywords = extract_keywords_tfidf(corpus[0], corpus, top_k=3)
    for word, score in keywords:
        print(f"  {word}: {score:.4f}")


# =============================================
# 课程总结
# =============================================
"""
核心收获：
- TF 衡量词在当前文档中的重要程度，常用方法有比率TF、对数TF和增强TF
- IDF 衡量词在整个语料库中的稀有程度，出现在越多文档中的词 IDF 越低
- TF-IDF = TF × IDF，自动平衡词频和稀有度，是最经典的关键词提取方法

常见陷阱：
- 不去除停用词就直接计算 TF-IDF，导致"的""是""了"等无意义词得分很高
- IDF 计算时忘记加平滑处理，当某个词不在任何文档中时会导致 log(0) 错误
- 用 TF-IDF 做文档相似度时忘记归一化，长文档的 TF 天然偏低会扭曲结果

下节课预告：
- 我们将学习条件随机场（CRF），一种能利用上下文信息进行序列标注的强大模型
"""
