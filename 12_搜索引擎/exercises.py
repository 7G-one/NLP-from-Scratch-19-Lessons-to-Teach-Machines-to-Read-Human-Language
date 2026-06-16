import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十二章：搜索引擎 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16
==============================================================================
"""


# ==============================================================================
# 练习 1：构建倒排索引
# ==============================================================================

def exercise_1_build_index(documents: list) -> dict:
    """
    练习 1：给定一组文档，构建倒排索引

    ━━━━━━━ 提示 ━━━━━━━
    1. 创建一个空字典 index = {}
    2. 遍历每个文档（用 enumerate 获取文档ID）
    3. 对每个文档，按空格分词
    4. 对每个词，把文档ID添加到该词的列表中
    5. 返回索引字典

    参数：
        documents: 文档列表，每个文档是空格分隔的词串
                   例如: ["机器 学习 算法", "深度 学习 模型"]

    返回：
        倒排索引字典 {词: [文档ID列表]}
        例如: {"机器": [0], "学习": [0, 1], "算法": [0], "深度": [1], "模型": [1]}
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # index = {}
    # for doc_id, doc in enumerate(documents):
    #     words = doc.split()
    #     for word in words:
    #         if word not in index:
    #             index[word] = []
    #         if doc_id not in index[word]:
    #             index[word].append(doc_id)
    # return index
    pass


def test_exercise_1():
    print("\n" + "=" * 60)
    print("练习 1：构建倒排索引")
    print("=" * 60)

    docs = ["机器 学习 算法", "深度 学习 模型", "机器 学习 模型"]
    result = exercise_1_build_index(docs)

    if result is None:
        print("[未完成] 请实现 exercise_1_build_index 函数")
        return False

    # 检查结果
    expected_keys = {"机器", "学习", "算法", "深度", "模型"}
    if set(result.keys()) != expected_keys:
        print(f"[错误] 期望键集合 {expected_keys}, 实际 {set(result.keys())}")
        return False

    if result.get("学习") == [0, 1, 2] and result.get("机器") == [0, 2]:
        print(f"[正确] 索引构建成功！")
        for word, doc_ids in sorted(result.items()):
            print(f"  '{word}' → {doc_ids}")
        return True
    else:
        print(f"[错误] 索引内容不正确: {result}")
        return False


# ==============================================================================
# 练习 2：计算 TF-IDF
# ==============================================================================

def exercise_2_tfidf(word_count_in_doc: int, doc_length: int,
                     total_docs: int, docs_with_word: int) -> float:
    """
    练习 2：计算单个词的 TF-IDF 分数

    ━━━━━━━ 提示 ━━━━━━━
    TF = word_count_in_doc / doc_length
    IDF = log(total_docs / (docs_with_word + 1)) + 1
    TF-IDF = TF * IDF

    可以用 math.log() 计算对数

    参数：
        word_count_in_doc: 词在当前文档中出现的次数
        doc_length: 当前文档的总词数
        total_docs: 语料库中的总文档数
        docs_with_word: 包含该词的文档数

    返回：
        TF-IDF 分数（浮点数）
    """
    import math

    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # tf = word_count_in_doc / doc_length
    # idf = math.log(total_docs / (docs_with_word + 1)) + 1
    # return tf * idf
    pass


def test_exercise_2():
    print("\n" + "=" * 60)
    print("练习 2：计算 TF-IDF")
    print("=" * 60)

    import math

    # 测试用例："机器" 在文档中出现2次，文档共10个词
    # 总共5个文档，其中2个文档包含"机器"
    result = exercise_2_tfidf(
        word_count_in_doc=2, doc_length=10,
        total_docs=5, docs_with_word=2
    )

    if result is None:
        print("[未完成] 请实现 exercise_2_tfidf 函数")
        return False

    expected_tf = 2 / 10
    expected_idf = math.log(5 / (2 + 1)) + 1
    expected = expected_tf * expected_idf

    if abs(result - expected) < 0.01:
        print(f"[正确] TF-IDF = {result:.4f}")
        print(f"  TF = {expected_tf}, IDF = {expected_idf:.4f}")
        return True
    else:
        print(f"[错误] 期望 {expected:.4f}, 实际 {result:.4f}")
        return False


# ==============================================================================
# 练习 3：布尔搜索
# ==============================================================================

def exercise_3_boolean_search(index: dict, query_words: list) -> list:
    """
    练习 3：实现布尔搜索（AND 逻辑）

    ━━━━━━━ 提示 ━━━━━━━
    1. 找到每个查询词对应的文档集合
    2. 取所有文档集合的交集（AND 逻辑）
    3. 返回排序后的文档ID列表

    参数：
        index: 倒排索引 {词: [文档ID列表]}
        query_words: 查询词列表

    返回：
        同时包含所有查询词的文档ID列表
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # if not query_words:
    #     return []
    # # 获取第一个词的文档集合
    # result = set(index.get(query_words[0], []))
    # # 取交集
    # for word in query_words[1:]:
    #     result &= set(index.get(word, []))
    # return sorted(result)
    pass


def test_exercise_3():
    print("\n" + "=" * 60)
    print("练习 3：布尔搜索")
    print("=" * 60)

    index = {
        "机器": [0, 2],
        "学习": [0, 1, 2],
        "深度": [1],
        "模型": [1, 2],
        "算法": [0],
    }

    test_cases = [
        (["机器", "学习"], [0, 2]),
        (["深度", "学习"], [1]),
        (["机器", "深度"], []),
        (["学习", "模型"], [1, 2]),
    ]

    all_correct = True
    for query, expected in test_cases:
        result = exercise_3_boolean_search(index, query)
        if result is None:
            print("[未完成] 请实现 exercise_3_boolean_search 函数")
            return False
        if result == expected:
            print(f"  [正确] 搜索 {query} → {result}")
        else:
            print(f"  [错误] 搜索 {query} → 期望 {expected}, 实际 {result}")
            all_correct = False

    return all_correct


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          G-one NLP 学院 - 第十二章 练习                    ║
    ║          搜索引擎                                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    results = []
    results.append(("练习1: 构建倒排索引", test_exercise_1()))
    results.append(("练习2: 计算TF-IDF", test_exercise_2()))
    results.append(("练习3: 布尔搜索", test_exercise_3()))

    print("\n" + "=" * 60)
    print("  练习清单")
    print("=" * 60)
    for name, passed in results:
        status = "[完成]" if passed else "[未完成]"
        print(f"  {status} {name}")

    completed = sum(1 for _, p in results if p)
    print(f"\n  完成率: {completed}/{len(results)}")
