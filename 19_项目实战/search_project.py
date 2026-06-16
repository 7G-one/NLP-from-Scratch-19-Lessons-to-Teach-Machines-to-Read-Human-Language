"""
==============================================================================
第十九章：项目实战 — 搜索引擎
==============================================================================
日期：2026-05-16

同学们好！这个模块我们来实现一个"完整的搜索引擎"。

----------------------------------------------------------------------
生活类比：搜索引擎就像"图书馆的检索系统"
----------------------------------------------------------------------

想象你在一个大型图书馆里，想找一本关于"机器学习"的书：

  1. 你走到检索机前，输入"机器学习"
  2. 检索机在"索引"中查找包含这些词的书
  3. 找到了 50 本相关的书
  4. 按"相关度"排序，最相关的排在最前面
  5. 显示前 10 本给你

搜索引擎的工作原理也是一样：
  1. 建立索引（预先扫描所有文档）
  2. 用户输入查询
  3. 在索引中查找匹配的文档
  4. 按相关度排序
  5. 返回结果

----------------------------------------------------------------------
本模块内容
----------------------------------------------------------------------

1. 倒排索引（Inverted Index）
2. TF-IDF 排序
3. 查询处理
4. 完整的搜索引擎实现

==============================================================================
"""

import math
import re
from collections import Counter, defaultdict


# ==============================================================================
# 第一部分：倒排索引（Inverted Index）
# ==============================================================================
#
# 倒排索引是搜索引擎的核心数据结构。
#
# 核心思想：
#   正向索引：文档 → 包含哪些词（文档的角度）
#   倒排索引：词 → 出现在哪些文档中（词的角度）
#
# 生活类比：
#   想象一本书的"索引"（在书的最后几页）：
#   - "机器学习" → 第 3, 15, 28 页
#   - "深度学习" → 第 10, 22, 35 页
#   - "神经网络" → 第 12, 25, 40 页
#
#   这就是倒排索引！
#
# ==============================================================================


class InvertedIndex:
    """
    倒排索引

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一本书的"索引页"：
    - 输入一个词
    - 快速找到包含这个词的所有文档
    """

    def __init__(self):
        """初始化倒排索引"""
        # 倒排索引：{词: [(文档ID, 词频), ...]}
        self.index = defaultdict(list)
        # 文档存储：{文档ID: 文档内容}
        self.documents = {}
        # 文档统计信息
        self.doc_lengths = {}  # 每篇文档的长度
        self.doc_count = 0     # 文档总数
        self.avg_doc_length = 0  # 平均文档长度

    def tokenize(self, text: str) -> list:
        """
        分词（简单的按字符分词）

        参数：
            text: 输入文本

        返回：
            词列表
        """
        tokens = []
        for char in text:
            if char.strip() and char not in "，。！？、；：""''（）【】《》\n\r\t":
                tokens.append(char)
        return tokens

    def add_document(self, doc_id: int, text: str):
        """
        添加文档到索引

        ━━━━━━━ 生活类比 ━━━━━━━
        就像图书馆新到了一本书，要在索引中登记：
        - 把书中的每个关键词都记录下来
        - 记录这个词在这本书中出现了多少次
        - 记录这本书在哪个书架上

        参数：
            doc_id: 文档 ID
            text: 文档内容
        """
        # 存储文档
        self.documents[doc_id] = text

        # 分词
        tokens = self.tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1

        # 统计词频
        term_freq = Counter(tokens)

        # 更新倒排索引
        for term, freq in term_freq.items():
            self.index[term].append((doc_id, freq))

        # 更新平均文档长度
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0

    def search(self, query: str) -> list:
        """
        搜索包含查询词的文档

        ━━━━━━━ 生活类比 ━━━━━━━
        就像在图书馆的索引中查找：
        - 输入"机器学习"
        - 找到所有包含"机器"和"学习"的文档
        - 返回文档列表

        参数：
            query: 查询文本

        返回：
            包含查询词的文档 ID 列表
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # 获取每个查询词对应的文档集合
        doc_sets = []
        for token in query_tokens:
            doc_ids = set(doc_id for doc_id, _ in self.index.get(token, []))
            doc_sets.append(doc_ids)

        if not doc_sets:
            return []

        # 取交集（所有查询词都必须出现）
        result_docs = doc_sets[0]
        for ds in doc_sets[1:]:
            result_docs = result_docs & ds

        return list(result_docs)


# ==============================================================================
# 第二部分：TF-IDF 排序
# ==============================================================================


class TFIDFScorer:
    """
    TF-IDF 评分器

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一个"裁判"，给每篇文档打分：
    - 如果文档中"机器学习"出现很多次 → TF 高 → 得分高
    - 如果"机器学习"在其他文档中很少出现 → IDF 高 → 得分高
    - 最终得分 = TF × IDF
    """

    def __init__(self, index: InvertedIndex):
        """
        初始化 TF-IDF 评分器

        参数：
            index: 倒排索引
        """
        self.index = index

    def compute_tf(self, term: str, doc_id: int) -> float:
        """
        计算词频（TF）

        公式：TF = 词在文档中出现的次数 / 文档总词数
        """
        doc_text = self.index.documents.get(doc_id, "")
        tokens = self.index.tokenize(doc_text)
        if not tokens:
            return 0
        count = tokens.count(term)
        return count / len(tokens)

    def compute_idf(self, term: str) -> float:
        """
        计算逆文档频率（IDF）

        公式：IDF = log(总文档数 / 包含该词的文档数)
        """
        doc_count = self.index.doc_count
        term_docs = self.index.index.get(term, [])
        df = len(term_docs)
        if df == 0:
            return 0
        return math.log((doc_count + 1) / (df + 1)) + 1

    def score(self, query: str, doc_id: int) -> float:
        """
        计算文档相对于查询的 TF-IDF 得分

        参数：
            query: 查询文本
            doc_id: 文档 ID

        返回：
            TF-IDF 得分
        """
        query_tokens = self.index.tokenize(query)
        total_score = 0

        for token in query_tokens:
            tf = self.compute_tf(token, doc_id)
            idf = self.compute_idf(token)
            total_score += tf * idf

        return total_score


# ==============================================================================
# 第三部分：BM25 排序
# ==============================================================================


class BM25Scorer:
    """
    BM25 评分器 —— 比 TF-IDF 更先进的排序算法

    ━━━━━━━ 生活类比 ━━━━━━━
    BM25 就像一个"更聪明的裁判"：
    - 不仅看词频，还考虑文档长度
    - 长文档的词频自然高，要适当"惩罚"
    - 短文档的词频低，要适当"奖励"

    BM25 是现代搜索引擎（如 Elasticsearch）的核心算法。
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 评分器

        参数：
            index: 倒排索引
            k1: 词频饱和参数（控制词频的影响程度）
            b: 文档长度归一化参数（控制文档长度的影响程度）
        """
        self.index = index
        self.k1 = k1
        self.b = b

    def score(self, query: str, doc_id: int) -> float:
        """
        计算 BM25 得分

        ━━━━━━━ 公式 ━━━━━━━
        BM25(q, d) = Σ IDF(t) × (f(t,d) × (k1 + 1)) / (f(t,d) + k1 × (1 - b + b × |d|/avgdl))

        其中：
        - IDF(t) = 词 t 的逆文档频率
        - f(t,d) = 词 t 在文档 d 中的词频
        - |d| = 文档 d 的长度
        - avgdl = 平均文档长度
        - k1, b = 调节参数

        参数：
            query: 查询文本
            doc_id: 文档 ID

        返回：
            BM25 得分
        """
        query_tokens = self.index.tokenize(query)
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        avg_length = self.index.avg_doc_length

        if avg_length == 0:
            return 0

        # 统计文档中每个词的频率
        doc_text = self.index.documents.get(doc_id, "")
        doc_tokens = self.index.tokenize(doc_text)
        term_freq = Counter(doc_tokens)

        score = 0
        for token in query_tokens:
            # IDF
            df = len(self.index.index.get(token, []))
            n = self.index.doc_count
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            # TF（带长度归一化）
            tf = term_freq.get(token, 0)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / avg_length))

            score += idf * tf_norm

        return score


# ==============================================================================
# 第四部分：完整的搜索引擎
# ==============================================================================


class SearchEngine:
    """
    完整的搜索引擎

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一个"智能图书馆检索系统"：
    1. 建立索引（扫描所有书籍）
    2. 用户输入查询
    3. 在索引中查找
    4. 按相关度排序
    5. 返回结果
    """

    def __init__(self):
        """初始化搜索引擎"""
        self.index = InvertedIndex()
        self.tfidf_scorer = TFIDFScorer(self.index)
        self.bm25_scorer = BM25Scorer(self.index)

    def add_document(self, doc_id: int, text: str):
        """
        添加文档

        参数：
            doc_id: 文档 ID
            text: 文档内容
        """
        self.index.add_document(doc_id, text)

    def search(self, query: str, top_k: int = 5, method: str = 'bm25') -> list:
        """
        搜索文档

        ━━━━━━━ 流程 ━━━━━━━
        1. 在倒排索引中查找包含查询词的文档
        2. 对每篇文档计算相关度得分
        3. 按得分降序排列
        4. 返回前 top_k 篇

        参数：
            query: 查询文本
            top_k: 返回前 k 个结果
            method: 排序方法 ('tfidf' 或 'bm25')

        返回：
            结果列表，每个元素是 (文档ID, 得分, 文档内容)
        """
        # 第一步：在索引中查找
        candidate_docs = self.index.search(query)

        if not candidate_docs:
            return []

        # 第二步：计算每篇文档的得分
        scorer = self.bm25_scorer if method == 'bm25' else self.tfidf_scorer
        scored_docs = []
        for doc_id in candidate_docs:
            score = scorer.score(query, doc_id)
            doc_text = self.index.documents[doc_id]
            scored_docs.append((doc_id, score, doc_text))

        # 第三步：按得分降序排列
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 第四步：返回前 top_k 个
        return scored_docs[:top_k]

    def highlight(self, text: str, query: str) -> str:
        """
        高亮显示查询词

        参数：
            text: 文档文本
            query: 查询文本

        返回：
            高亮后的文本
        """
        query_tokens = self.index.tokenize(query)
        highlighted = text
        for token in query_tokens:
            highlighted = highlighted.replace(token, f"[{token}]")
        return highlighted


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十九章                      ║
    ║        项目实战：搜索引擎                             ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 创建搜索引擎
    engine = SearchEngine()

    # 添加文档
    documents = {
        0: "机器学习是人工智能的重要分支。它让计算机能够从数据中自动学习。",
        1: "深度学习是机器学习的一个热门方向。它使用多层神经网络。",
        2: "自然语言处理让计算机理解和生成人类语言。分词是基础任务。",
        3: "卷积神经网络在图像识别领域取得了巨大成功。",
        4: "循环神经网络适合处理序列数据，如文本和语音。",
        5: "今天股市大涨，上证指数突破三千点。",
        6: "央行宣布降息，刺激经济增长。",
        7: "Python 是最流行的编程语言之一，广泛用于数据科学。",
    }

    print("  添加文档:")
    for doc_id, text in documents.items():
        engine.add_document(doc_id, text)
        print(f"    [{doc_id}] {text[:30]}...")

    # 搜索演示
    queries = ["机器学习", "神经网络", "自然语言处理", "股市"]

    print("\n" + "=" * 60)
    print("  搜索结果")
    print("=" * 60)

    for query in queries:
        print(f"\n  查询: '{query}'")
        results = engine.search(query, top_k=3)

        if results:
            for doc_id, score, text in results:
                highlighted = engine.highlight(text, query)
                print(f"    [{doc_id}] 得分: {score:.4f}")
                print(f"         {highlighted[:50]}...")
        else:
            print("    未找到相关文档")

    # BM25 vs TF-IDF 对比
    print("\n" + "=" * 60)
    print("  BM25 vs TF-IDF 对比")
    print("=" * 60)

    query = "机器学习深度学习"
    print(f"\n  查询: '{query}'")

    results_tfidf = engine.search(query, top_k=3, method='tfidf')
    results_bm25 = engine.search(query, top_k=3, method='bm25')

    print("\n  TF-IDF 排序:")
    for doc_id, score, text in results_tfidf:
        print(f"    [{doc_id}] {score:.4f} - {text[:30]}...")

    print("\n  BM25 排序:")
    for doc_id, score, text in results_bm25:
        print(f"    [{doc_id}] {score:.4f} - {text[:30]}...")
