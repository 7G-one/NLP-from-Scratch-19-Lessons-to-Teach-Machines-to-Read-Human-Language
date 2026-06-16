"""
==============================================================================
第十二章：搜索引擎（Search Engine）
==============================================================================
日期：2026-05-16

同学们好！前面的章节我们学了分词、词性标注、TF-IDF 等基础技术，
今天我们把这些技术组合起来，构建一个完整的搜索引擎！

----------------------------------------------------------------------
生活类比：搜索引擎就像图书馆的索引系统
----------------------------------------------------------------------

想象你走进一个巨大的图书馆，想找关于"机器学习"的书：

  方案一（暴力搜索）：
    从第一排书架开始，一本一本翻，看目录里有没有"机器学习"。
    → 效率极低！如果有 100 万本书，你可能要翻一整年。

  方案二（索引系统）：
    图书馆有一套索引卡片系统：
    ┌──────────────────────────────────────────────┐
    │  "机器学习" → 第3排第5架、第7排第2架、...       │
    │  "深度学习" → 第5排第1架、第7排第2架、...       │
    │  "自然语言" → 第2排第3架、第7排第2架、...       │
    └──────────────────────────────────────────────┘
    直接查卡片就知道书在哪里！→ 效率极高！

搜索引擎的核心就是这个"索引系统"——在 NLP 中叫"倒排索引"（Inverted Index）。

==============================================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import math
from collections import defaultdict


# ==============================================================================
# 第一部分：文档预处理
# ==============================================================================
#
# 在建索引之前，需要对文档进行预处理：
# 1. 中文分词（把句子切成词）
# 2. 去停用词（去掉"的"、"了"、"是"等无意义的词）
# 3. 统一转小写（英文场景）
#
# 就像整理图书馆：先把书分类、编号，才能建索引。
#
# ==============================================================================

# 简化的停用词表
# 这些词出现频率极高，但对搜索没有帮助
STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就",
    "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "他", "她", "它",
    "吗", "吧", "呢", "啊", "哦", "嗯",
    "the", "a", "an", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "with",
}


def simple_tokenize(text: str) -> list:
    """
    简单的中文分词（按字符级别切分 + 简单词典匹配）

    ━━━━━━━ 生活类比 ━━━━━━━
    分词就像切菜：把一整块食材切成小块，方便后续烹饪。
    搜索引擎需要先把句子切成词，才能建立索引。

    参数：
        text: 输入文本

    返回：
        词列表
    """
    # 简单的词典（实际搜索引擎会用 jieba 等专业分词工具）
    known_words = {
        "机器学习", "深度学习", "自然语言处理", "搜索引擎",
        "人工智能", "计算机", "算法", "模型", "数据", "网络",
        "学习", "搜索", "处理", "分析", "训练", "模型",
        "文本", "分类", "聚类", "推荐", "系统", "技术",
        "中国", "北京", "上海", "清华大学", "北京大学",
        "很好", "非常", "已经", "可以", "能够",
    }

    words = []
    i = 0
    while i < len(text):
        # 尝试最长匹配（贪心策略）
        matched = False
        for length in range(min(6, len(text) - i), 0, -1):
            candidate = text[i:i + length]
            if candidate in known_words:
                words.append(candidate)
                i += length
                matched = True
                break

        if not matched:
            # 单字作为一个词
            char = text[i]
            if char.strip():  # 跳过空白字符
                words.append(char)
            i += 1

    return words


def preprocess(text: str) -> list:
    """
    文档预处理：分词 + 去停用词

    参数：
        text: 原始文本

    返回：
        处理后的词列表
    """
    words = simple_tokenize(text)
    # 去掉停用词
    return [w for w in words if w not in STOP_WORDS]


# ==============================================================================
# 第二部分：倒排索引（Inverted Index）
# ==============================================================================
#
# 倒排索引是搜索引擎的核心数据结构！
#
# 正排索引（正向索引）：文档 → 包含哪些词
#   文档1: ["机器", "学习", "算法"]
#   文档2: ["深度", "学习", "模型"]
#
# 倒排索引：词 → 出现在哪些文档中
#   "机器" → [文档1]
#   "学习" → [文档1, 文档2]
#   "算法" → [文档1]
#   "深度" → [文档2]
#   "模型" → [文档2]
#
# 就像图书馆的索引卡片：
#   正排索引 = 每本书的目录（这本书有什么内容）
#   倒排索引 = 主题卡片柜（这个主题有哪些书）
#
# ==============================================================================

class InvertedIndex:
    """
    倒排索引实现

    ━━━━━━━ 核心思想 ━━━━━━━
    倒排索引 = {词: [包含该词的文档ID列表]}

    还会记录每个词在每个文档中出现的频率（TF），
    这样后续可以用 TF-IDF 进行排序。

    ━━━━━━━ 数据结构 ━━━━━━━
    {
        "机器学习": {
            doc_id_1: [位置1, 位置3],    # 在文档1的第1和第3个位置出现
            doc_id_2: [位置0],           # 在文档2的第0个位置出现
        },
        "算法": {
            doc_id_1: [位置2],
        },
    }
    """

    def __init__(self):
        """初始化倒排索引"""
        # 核心数据结构：词 → {文档ID: 词频}
        self.index = defaultdict(lambda: defaultdict(int))
        # 文档存储：文档ID → 原始文本
        self.documents = {}
        # 文档ID计数器
        self.doc_count = 0
        # 每个文档的词数（用于计算TF）
        self.doc_lengths = {}

    def add_document(self, text: str) -> int:
        """
        添加文档到索引

        ━━━━━━━ 生活类比 ━━━━━━━
        就像图书馆新到了一批书，要：
        1. 给书编号（分配文档ID）
        2. 看看书里有哪些关键词
        3. 更新索引卡片

        参数：
            text: 文档文本

        返回：
            文档ID
        """
        doc_id = self.doc_count
        self.doc_count += 1

        # 存储原始文档
        self.documents[doc_id] = text

        # 预处理：分词 + 去停用词
        words = preprocess(text)
        self.doc_lengths[doc_id] = len(words)

        # 更新倒排索引
        for position, word in enumerate(words):
            self.index[word][doc_id] += 1

        return doc_id

    def search(self, query: str) -> list:
        """
        搜索文档（布尔检索模型）

        ━━━━━━━ 生活类比 ━━━━━━━
        就像在图书馆查索引卡片：
        - 你想要找关于"机器学习"的书
        - 翻到"机器学习"的卡片，上面列出了所有相关书籍
        - 如果搜多个词，找同时出现在多个卡片下的书

        参数：
            query: 查询文本

        返回：
            匹配的文档ID列表
        """
        query_words = preprocess(query)

        if not query_words:
            return []

        # 找到包含所有查询词的文档（AND 逻辑）
        # 先找第一个词的文档集合
        first_word = query_words[0]
        if first_word not in self.index:
            return []

        result_docs = set(self.index[first_word].keys())

        # 取交集（AND 逻辑）
        for word in query_words[1:]:
            if word not in self.index:
                return []  # 有一个词不在索引中，结果为空
            result_docs &= set(self.index[word].keys())

        return sorted(result_docs)

    def get_term_frequency(self, word: str, doc_id: int) -> int:
        """获取词在文档中的出现次数"""
        return self.index[word][doc_id]

    def get_document_frequency(self, word: str) -> int:
        """获取包含该词的文档数量"""
        return len(self.index[word])

    def display_index(self):
        """打印倒排索引的内容（用于教学演示）"""
        print("\n倒排索引内容：")
        print("-" * 50)
        for word in sorted(self.index.keys()):
            doc_ids = list(self.index[word].keys())
            print(f"  '{word}' → 文档 {doc_ids}")


# ==============================================================================
# 第三部分：TF-IDF 搜索排序
# ==============================================================================
#
# 布尔搜索只告诉你"哪些文档包含这些词"，但不告诉你"哪个文档更相关"。
#
# TF-IDF 可以帮我们解决这个问题！
#
# TF（词频）= 这个词在文档中出现的次数
#   → 出现次数越多，越可能相关
#
# IDF（逆文档频率）= log(总文档数 / 包含该词的文档数)
#   → 越稀有的词，区分度越高
#
# 生活类比：
#   想象你在一个班级里找"最像程序员"的同学：
#   - "穿格子衫"（TF高：他穿了3件格子衫）→ 可能是程序员
#   - "有两只眼睛"（IDF低：大家都有）→ 没区分度
#   - "用 Vim 编辑器"（IDF高：只有他用）→ 很可能是程序员！
#
# ==============================================================================

class TFIDFSearchEngine:
    """
    基于 TF-IDF 的搜索引擎

    ━━━━━━━ 工作流程 ━━━━━━━
    1. 用户输入查询词
    2. 用倒排索引快速找到候选文档
    3. 用 TF-IDF 计算每个文档的相关度分数
    4. 按分数排序，返回结果
    """

    def __init__(self):
        """初始化搜索引擎"""
        self.index = InvertedIndex()

    def add_document(self, text: str) -> int:
        """添加文档"""
        return self.index.add_document(text)

    def compute_tfidf(self, word: str, doc_id: int) -> float:
        """
        计算单个词在单个文档中的 TF-IDF 分数

        ━━━━━━━ 公式 ━━━━━━━
        TF-IDF = TF × IDF

        TF（词频）= 词在文档中出现的次数 / 文档总词数
        IDF（逆文档频率）= log(总文档数 / 包含该词的文档数 + 1)

        参数：
            word: 查询词
            doc_id: 文档ID

        返回：
            TF-IDF 分数
        """
        # 获取词频
        tf = self.index.get_term_frequency(word, doc_id)
        doc_len = self.index.doc_lengths.get(doc_id, 1)

        # 归一化TF：词频 / 文档总词数
        tf_normalized = tf / doc_len if doc_len > 0 else 0

        # 计算IDF
        N = self.index.doc_count  # 总文档数
        df = self.index.get_document_frequency(word)  # 包含该词的文档数
        idf = math.log(N / (df + 1)) + 1  # +1 平滑，避免除零

        return tf_normalized * idf

    def search(self, query: str, top_k: int = 5) -> list:
        """
        搜索并按 TF-IDF 分数排序

        ━━━━━━━ 生活类比 ━━━━━━━
        就像面试筛选简历：
        1. 先用关键词筛选出候选人（倒排索引）
        2. 再给每个候选人打分（TF-IDF）
        3. 按分数排序，取前几名（top-k）

        参数：
            query: 查询文本
            top_k: 返回前k个结果

        返回：
            [(文档ID, 分数, 文本), ...] 按分数降序排列
        """
        query_words = preprocess(query)

        if not query_words:
            return []

        # 用倒排索引找到候选文档
        candidate_docs = set()
        for word in query_words:
            if word in self.index.index:
                candidate_docs.update(self.index.index[word].keys())

        if not candidate_docs:
            return []

        # 计算每个文档的总TF-IDF分数
        scores = {}
        for doc_id in candidate_docs:
            total_score = 0
            for word in query_words:
                total_score += self.compute_tfidf(word, doc_id)
            scores[doc_id] = total_score

        # 按分数降序排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回 top-k 结果
        results = []
        for doc_id, score in ranked[:top_k]:
            results.append((doc_id, score, self.index.documents[doc_id]))

        return results


# ==============================================================================
# 第四部分：向量空间模型（VSM）搜索
# ==============================================================================
#
# TF-IDF 还有一个更高级的用法：把文档表示为向量！
#
# 假设我们的词表有 5 个词：["机器", "学习", "深度", "算法", "数据"]
# 那么每个文档可以表示为一个 5 维向量：
#   文档1 "机器学习算法" → [1, 1, 0, 1, 0]
#   文档2 "深度学习数据" → [0, 1, 1, 0, 1]
#
# 查询也可以表示为向量：
#   查询 "学习" → [0, 1, 0, 0, 0]
#
# 然后用余弦相似度计算查询和文档的相似程度！
#
# 生活类比：
#   就像比较两个人的兴趣爱好：
#   - 小明喜欢：[音乐=8, 电影=6, 运动=9, 游戏=2]
#   - 小红喜欢：[音乐=7, 电影=8, 运动=3, 游戏=1]
#   - 小刚喜欢：[音乐=1, 电影=2, 运动=8, 游戏=9]
#   → 小明和小红更相似！（余弦相似度更高）
#
# ==============================================================================

class VectorSearchEngine:
    """
    基于向量空间模型的搜索引擎

    使用 TF-IDF 向量 + 余弦相似度进行文档排序
    """

    def __init__(self):
        """初始化向量搜索引擎"""
        self.index = InvertedIndex()
        self.tfidf_engine = TFIDFSearchEngine()
        # 词表（所有不重复的词）
        self.vocabulary = []
        self.vocab_to_idx = {}

    def add_document(self, text: str) -> int:
        """添加文档"""
        doc_id = self.index.add_document(text)
        self.tfidf_engine.index = self.index
        # 更新词表
        self._build_vocabulary()
        return doc_id

    def _build_vocabulary(self):
        """构建词表"""
        self.vocabulary = sorted(self.index.index.keys())
        self.vocab_to_idx = {w: i for i, w in enumerate(self.vocabulary)}

    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        将文本转换为 TF-IDF 向量

        参数：
            text: 输入文本

        返回：
            numpy 数组（TF-IDF 向量）
        """
        words = preprocess(text)
        vec = np.zeros(len(self.vocabulary))

        # 统计词频
        word_counts = defaultdict(int)
        for w in words:
            word_counts[w] += 1

        # 计算TF-IDF
        N = self.index.doc_count
        for word, count in word_counts.items():
            if word in self.vocab_to_idx:
                idx = self.vocab_to_idx[word]
                df = self.index.get_document_frequency(word)
                idf = math.log(N / (df + 1)) + 1
                tf = count / len(words) if words else 0
                vec[idx] = tf * idf

        return vec

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度

        ━━━━━━━ 公式 ━━━━━━━
        cos(θ) = (A · B) / (|A| × |B|)

        结果范围：[-1, 1]
        - 1 表示完全相同
        - 0 表示完全无关
        - -1 表示完全相反
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 5) -> list:
        """
        向量空间模型搜索

        参数：
            query: 查询文本
            top_k: 返回前k个结果

        返回：
            [(文档ID, 相似度, 文本), ...]
        """
        # 先用倒排索引缩小候选范围
        candidate_docs = self.index.search(query)

        if not candidate_docs:
            # 如果倒排索引没找到，尝试所有文档
            candidate_docs = list(self.index.documents.keys())

        # 计算查询向量
        query_vec = self._text_to_vector(query)

        # 计算每个候选文档的相似度
        scores = []
        for doc_id in candidate_docs:
            doc_vec = self._text_to_vector(self.index.documents[doc_id])
            sim = self._cosine_similarity(query_vec, doc_vec)
            scores.append((doc_id, sim))

        # 按相似度降序排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 top-k
        results = []
        for doc_id, sim in scores[:top_k]:
            results.append((doc_id, sim, self.index.documents[doc_id]))

        return results


# ==============================================================================
# 第五部分：Elasticsearch 概念介绍
# ==============================================================================
#
# 我们刚才实现的是一个"迷你版"搜索引擎。
# 在工业界，最常用的搜索引擎是 Elasticsearch（简称 ES）。
#
# Elasticsearch 是什么？
#   - 一个分布式的全文搜索引擎
#   - 基于 Apache Lucene 构建
#   - 支持海量数据的实时搜索
#   - 被广泛用于日志分析、电商搜索、推荐系统等
#
# 核心概念（对应我们实现的组件）：
#
#   我们的实现          │  Elasticsearch
#   ───────────────────┼──────────────────
#   InvertedIndex      │  Shard（分片）
#   add_document()     │  Index API（索引文档）
#   search()           │  Search API（搜索）
#   TF-IDF             │  BM25（ES默认的评分算法）
#   preprocess()       │  Analyzer（分析器）
#
# ==============================================================================

def explain_elasticsearch():
    """
    Elasticsearch 核心概念介绍

    这个函数以教学的方式解释 Elasticsearch 的核心概念，
    不需要安装 ES，纯文字说明。
    """
    concepts = {
        "Index（索引）": """
        索引就像数据库中的"表"，是文档的集合。
        例如：一个电商网站可能有 "products" 索引、"users" 索引。
        我们的 InvertedIndex 类就是一个简化版的索引。
        """,

        "Document（文档）": """
        文档是索引中的基本单位，用 JSON 格式存储。
        例如：{"name": "iPhone 15", "price": 7999, "category": "手机"}
        我们的 add_document() 方法就是在添加文档。
        """,

        "Mapping（映射）": """
        映射定义了文档中每个字段的类型和处理方式。
        例如：name 是 text 类型（需要分词），price 是 number 类型。
        就像数据库的 schema。
        """,

        "Analyzer（分析器）": """
        分析器负责把文本切成词（Token）。
        包含三步：
        1. Character Filter：预处理（如去掉HTML标签）
        2. Tokenizer：分词（如按空格切、按n-gram切）
        3. Token Filter：过滤（如转小写、去停用词）
        我们的 preprocess() 函数就是一个简化版的分析器。
        """,

        "BM25（评分算法）": """
        BM25 是 Elasticsearch 默认的文档评分算法，
        可以理解为 TF-IDF 的"升级版"。
        主要改进：
        1. TF 有饱和效应（词出现10次和100次差别不大）
        2. 考虑了文档长度的影响
        我们用的 TF-IDF 是 BM25 的简化版。
        """,

        "Shard（分片）": """
        分片是把索引拆分成多个部分，分布在不同的服务器上。
        这样可以并行搜索，提高速度。
        就像图书馆有多个分馆，每个分馆管理一部分书籍。
        """,
    }

    return concepts


# ==============================================================================
# 第六部分：简单搜索引擎完整实现
# ==============================================================================

class SimpleSearchEngine:
    """
    简单搜索引擎 —— 整合所有功能

    ━━━━━━━ 完整工作流程 ━━━━━━━

    1. 建立索引（Indexing）：
       输入：文档集合
       过程：分词 → 去停用词 → 建立倒排索引

    2. 搜索查询（Searching）：
       输入：用户查询
       过程：分词 → 查倒排索引 → TF-IDF 评分 → 排序 → 返回结果
    """

    def __init__(self):
        """初始化搜索引擎"""
        self.tfidf_engine = TFIDFSearchEngine()
        self.vector_engine = VectorSearchEngine()

    def index_documents(self, documents: list):
        """
        批量索引文档

        参数：
            documents: 文档文本列表
        """
        for text in documents:
            self.tfidf_engine.add_document(text)
            self.vector_engine.add_document(text)
        print(f"  已索引 {len(documents)} 篇文档")

    def search(self, query: str, method: str = "tfidf", top_k: int = 3) -> list:
        """
        搜索文档

        参数：
            query: 查询文本
            method: "tfidf" 或 "vector"
            top_k: 返回前k个结果

        返回：
            搜索结果列表
        """
        if method == "tfidf":
            return self.tfidf_engine.search(query, top_k)
        elif method == "vector":
            return self.vector_engine.search(query, top_k)
        else:
            raise ValueError(f"未知的搜索方法: {method}")

    def display_results(self, query: str, results: list):
        """
        美观地显示搜索结果

        参数：
            query: 查询文本
            results: 搜索结果
        """
        print(f"\n  查询: '{query}'")
        print(f"  找到 {len(results)} 条结果：")
        print("  " + "-" * 50)

        for i, item in enumerate(results):
            doc_id, score, text = item
            # 截断显示（太长的文档只显示前50个字符）
            display_text = text[:50] + "..." if len(text) > 50 else text
            print(f"  [{i + 1}] 文档{doc_id} (分数: {score:.4f})")
            print(f"      {display_text}")


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_inverted_index():
    """演示倒排索引"""
    print("=" * 60)
    print("倒排索引演示")
    print("=" * 60)

    index = InvertedIndex()

    # 添加文档
    docs = [
        "机器学习是人工智能的核心技术",
        "深度学习是机器学习的一个分支",
        "自然语言处理是人工智能的重要应用",
        "搜索引擎使用了倒排索引技术",
    ]

    for doc in docs:
        doc_id = index.add_document(doc)
        print(f"  添加文档 {doc_id}: {doc}")

    # 显示索引
    index.display_index()

    # 搜索
    queries = ["机器学习", "人工智能", "搜索引擎"]
    for q in queries:
        results = index.search(q)
        print(f"\n  搜索 '{q}' → 文档 {results}")


def demo_tfidf_search():
    """演示 TF-IDF 搜索"""
    print("\n" + "=" * 60)
    print("TF-IDF 搜索演示")
    print("=" * 60)

    engine = TFIDFSearchEngine()

    docs = [
        "机器学习是人工智能的核心技术，机器学习需要大量数据",
        "深度学习是机器学习的一个重要分支",
        "自然语言处理是人工智能的重要应用领域",
        "搜索引擎使用倒排索引和TF-IDF技术",
        "机器学习算法包括决策树和支持向量机",
    ]

    for doc in docs:
        engine.add_document(doc)

    query = "机器学习"
    results = engine.search(query)
    print(f"\n  查询: '{query}'")
    for doc_id, score, text in results:
        print(f"  文档{doc_id} (分数: {score:.4f}): {text}")


def demo_vector_search():
    """演示向量空间搜索"""
    print("\n" + "=" * 60)
    print("向量空间模型搜索演示")
    print("=" * 60)

    engine = VectorSearchEngine()

    docs = [
        "机器学习算法需要大量训练数据",
        "深度学习使用神经网络模型",
        "自然语言处理研究文本分析技术",
        "搜索引擎的核心是倒排索引",
    ]

    for doc in docs:
        engine.add_document(doc)

    query = "深度学习模型"
    results = engine.search(query)
    print(f"\n  查询: '{query}'")
    for doc_id, sim, text in results:
        print(f"  文档{doc_id} (相似度: {sim:.4f}): {text}")


def demo_elasticsearch_concepts():
    """演示 Elasticsearch 概念"""
    print("\n" + "=" * 60)
    print("Elasticsearch 核心概念")
    print("=" * 60)

    concepts = explain_elasticsearch()
    for name, explanation in concepts.items():
        print(f"\n  【{name}】")
        for line in explanation.strip().split("\n"):
            print(f"    {line.strip()}")


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十二章                      ║
    ║        搜索引擎                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    demo_inverted_index()
    demo_tfidf_search()
    demo_vector_search()
    demo_elasticsearch_concepts()

    # 总结
    print("\n" + "=" * 60)
    print("第十二章 总结")
    print("=" * 60)
    print("""
    本章我们学习了搜索引擎的核心技术：

    [OK] 倒排索引 — 词 → 文档ID列表（搜索引擎的基础）
    [OK] TF-IDF 排序 — 用词频和逆文档频率计算相关度
    [OK] 向量空间模型 — 把文档变成向量，用余弦相似度匹配
    [OK] Elasticsearch 概念 — 业界搜索引擎的标准方案
    """)

    # =============================================
    # 下节课预告
    # =============================================
    """
    下节课我们将学习 Word2Vec 词向量：
    - 词向量的概念 —— 给每个词一个"数字身份证"
    - CBOW 与 Skip-gram —— Word2Vec 的两种训练方式
    - 词类比推理 —— 国王 - 男人 + 女人 ≈ 女王
    """
