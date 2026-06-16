import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十三章：Word2Vec 词向量 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16
==============================================================================
"""

try:
    import numpy as np
except ImportError:
    print("本课需要 numpy 库，请运行：pip install numpy")
    print("安装后重新运行本文件即可。")
    exit(1)


# ==============================================================================
# 练习 1：余弦相似度计算
# ==============================================================================

def exercise_1_cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    练习 1：计算两个向量的余弦相似度

    ━━━━━━━ 提示 ━━━━━━━
    余弦相似度公式：
        cos(θ) = (A · B) / (|A| × |B|)

    步骤：
    1. 计算点积 A · B（对应位置相乘再求和）
    2. 计算 |A|（向量A的模，各分量平方和开根号）
    3. 计算 |B|
    4. 返回 cos(θ) = dot / (|A| * |B|)

    可以用 np.dot(), np.linalg.norm()

    参数：
        vec_a: 向量A（列表）
        vec_b: 向量B（列表）

    返回：
        余弦相似度（浮点数，范围 [-1, 1]）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # a = np.array(vec_a)
    # b = np.array(vec_b)
    # dot = np.dot(a, b)
    # norm_a = np.linalg.norm(a)
    # norm_b = np.linalg.norm(b)
    # if norm_a == 0 or norm_b == 0:
    #     return 0.0
    # return dot / (norm_a * norm_b)
    pass


def test_exercise_1():
    print("\n" + "=" * 60)
    print("练习 1：余弦相似度计算")
    print("=" * 60)

    test_cases = [
        ([1, 0, 0], [0, 1, 0], 0.0),       # 正交向量
        ([1, 1, 1], [1, 1, 1], 1.0),       # 完全相同
        ([1, 0, 0], [-1, 0, 0], -1.0),     # 完全相反
        ([1, 2, 3], [1, 2, 3], 1.0),       # 完全相同
    ]

    all_correct = True
    for vec_a, vec_b, expected in test_cases:
        result = exercise_1_cosine_similarity(vec_a, vec_b)
        if result is None:
            print("[未完成] 请实现 exercise_1_cosine_similarity 函数")
            return False
        if abs(result - expected) < 0.01:
            print(f"  [正确] {vec_a} vs {vec_b} = {result:.4f}")
        else:
            print(f"  [错误] {vec_a} vs {vec_b} = 期望 {expected:.4f}, 实际 {result:.4f}")
            all_correct = False

    return all_correct


# ==============================================================================
# 练习 2：CBOW 前向传播
# ==============================================================================

def exercise_2_cbow_forward(embedding_matrix: np.ndarray,
                            context_indices: list) -> np.ndarray:
    """
    练习 2：实现 CBOW 的前向传播

    ━━━━━━━ 提示 ━━━━━━━
    CBOW 前向传播非常简单：
    1. 从 embedding_matrix 中取出上下文词的向量
    2. 对所有上下文词向量取平均
    3. 返回平均向量

    例如：
    embedding_matrix = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    context_indices = [0, 2]
    → 取出第0行和第2行: [[0.1, 0.2], [0.5, 0.6]]
    → 取平均: [0.3, 0.4]

    参数：
        embedding_matrix: 词向量矩阵 (vocab_size, embedding_dim)
        context_indices: 上下文词的索引列表

    返回：
        上下文词向量的平均值 (embedding_dim,)
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # context_vectors = embedding_matrix[context_indices]
    # return np.mean(context_vectors, axis=0)
    pass


def test_exercise_2():
    print("\n" + "=" * 60)
    print("练习 2：CBOW 前向传播")
    print("=" * 60)

    embedding_matrix = np.array([
        [0.1, 0.2, 0.3],  # 词0
        [0.4, 0.5, 0.6],  # 词1
        [0.7, 0.8, 0.9],  # 词2
        [1.0, 1.1, 1.2],  # 词3
    ])

    test_cases = [
        ([0, 2], [0.4, 0.5, 0.6]),     # (词0+词2)/2
        ([1, 3], [0.7, 0.8, 0.9]),     # (词1+词3)/2
        ([0, 1, 2], [0.4, 0.5, 0.6]), # (词0+词1+词2)/3
    ]

    all_correct = True
    for ctx_indices, expected in test_cases:
        result = exercise_2_cbow_forward(embedding_matrix, ctx_indices)
        if result is None:
            print("[未完成] 请实现 exercise_2_cbow_forward 函数")
            return False
        expected_arr = np.array(expected)
        if np.allclose(result, expected_arr, atol=0.01):
            print(f"  [正确] 上下文{ctx_indices} → {result}")
        else:
            print(f"  [错误] 上下文{ctx_indices} → 期望 {expected}, 实际 {result}")
            all_correct = False

    return all_correct


# ==============================================================================
# 练习 3：找最相似的词
# ==============================================================================

def exercise_3_most_similar(embedding_matrix: np.ndarray,
                            word_idx: int,
                            vocab: list,
                            top_k: int = 3) -> list:
    """
    练习 3：给定词向量矩阵和目标词索引，找最相似的 top_k 个词

    ━━━━━━━ 提示 ━━━━━━━
    1. 获取目标词的向量
    2. 遍历所有词，计算余弦相似度
    3. 排除目标词自身
    4. 按相似度降序排序
    5. 返回前 top_k 个 (词, 相似度)

    参数：
        embedding_matrix: 词向量矩阵 (vocab_size, embedding_dim)
        word_idx: 目标词的索引
        vocab: 词表列表
        top_k: 返回前k个结果

    返回：
        [(词, 相似度), ...] 列表
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # target_vec = embedding_matrix[word_idx]
    # similarities = []
    # for i in range(len(vocab)):
    #     if i == word_idx:
    #         continue
    #     vec = embedding_matrix[i]
    #     cos_sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10)
    #     similarities.append((vocab[i], cos_sim))
    # similarities.sort(key=lambda x: x[1], reverse=True)
    # return similarities[:top_k]
    pass


def test_exercise_3():
    print("\n" + "=" * 60)
    print("练习 3：找最相似的词")
    print("=" * 60)

    # 构造词向量：词0和词1相似，词2不同
    embedding_matrix = np.array([
        [1.0, 0.0, 0.0],  # 国王
        [0.9, 0.1, 0.0],  # 女王（和国王接近）
        [0.0, 0.0, 1.0],  # 苹果（和国王很远）
        [0.8, 0.2, 0.1],  # 皇帝（和国王接近）
    ])
    vocab = ["国王", "女王", "苹果", "皇帝"]

    result = exercise_3_most_similar(embedding_matrix, word_idx=0, vocab=vocab, top_k=2)

    if result is None:
        print("[未完成] 请实现 exercise_3_most_similar 函数")
        return False

    # 检查结果
    result_words = [w for w, s in result]
    if "女王" in result_words and "皇帝" in result_words:
        print(f"[正确] 与'国王'最相似的词: {result}")
        return True
    elif len(result) == 2:
        print(f"[部分正确] 结果: {result}")
        print(f"  期望'女王'和'皇帝'排在前两位")
        return False
    else:
        print(f"[错误] 结果: {result}")
        return False


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          G-one NLP 学院 - 第十三章 练习                    ║
    ║          Word2Vec 词向量                                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    results = []
    results.append(("练习1: 余弦相似度", test_exercise_1()))
    results.append(("练习2: CBOW前向传播", test_exercise_2()))
    results.append(("练习3: 找最相似的词", test_exercise_3()))

    print("\n" + "=" * 60)
    print("  练习清单")
    print("=" * 60)
    for name, passed in results:
        status = "[完成]" if passed else "[未完成]"
        print(f"  {status} {name}")

    completed = sum(1 for _, p in results if p)
    print(f"\n  完成率: {completed}/{len(results)}")
