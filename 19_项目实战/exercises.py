import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十九章：项目实战 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 实现简单的聊天机器人规则匹配
    2. 实现倒排索引
    3. 实现简单的协同过滤推荐

运行方式：
    python exercises.py

提示：
    - 每个练习都有详细的提示，按照提示一步步来
    - 先自己写，写不出来再看注释中的参考答案
    - 运行后会自动检查你的答案是否正确
==============================================================================
"""

import math
from collections import defaultdict, Counter


# ==============================================================================
# 练习 1：实现聊天机器人规则匹配
# ==============================================================================

def exercise_1_chatbot_response(user_input: str, rules: dict) -> str:
    """
    练习 1：实现简单的聊天机器人规则匹配

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你是一个客服，有一本"标准答案手册"：
    - 用户问"怎么退货" → 翻到"退货"那一页，照着念
    - 用户问"多少钱" → 翻到"价格"那一页，照着念

    ━━━━━━━ 提示 ━━━━━━━
    1. 遍历 rules 中的每个规则
       - rules 格式：{关键词: 回复}

    2. 检查 user_input 中是否包含关键词
       - 用 keyword in user_input 判断

    3. 如果找到匹配的关键词，返回对应的回复

    4. 如果没有匹配，返回默认回复 "我不太明白"

    参数：
        user_input: 用户输入
        rules: 规则字典 {关键词: 回复}

    返回：
        机器人的回复

    示例：
        >>> rules = {"你好": "你好呀！", "天气": "今天天气不错"}
        >>> exercise_1_chatbot_response("你好", rules)
        "你好呀！"
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # for keyword, response in rules.items():
    #     if keyword in user_input:
    #         return response
    # return "我不太明白"
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：聊天机器人规则匹配")
    print("=" * 60)

    rules = {
        "你好": "你好！有什么可以帮你的吗？",
        "天气": "今天天气不错！",
        "再见": "再见！祝你愉快！",
        "谢谢": "不客气！",
    }

    test_cases = [
        ("你好呀", "你好！有什么可以帮你的吗？"),
        ("今天天气怎么样", "今天天气不错！"),
        ("再见", "再见！祝你愉快！"),
        ("随便说点什么", "我不太明白"),
    ]

    all_correct = True
    for user_input, expected in test_cases:
        result = exercise_1_chatbot_response(user_input, rules)
        if result is None:
            print("[未完成] 请实现 exercise_1_chatbot_response 函数")
            return False
        if result == expected:
            print(f"  [正确] '{user_input}' → '{result}'")
        else:
            print(f"  [错误] '{user_input}' → '{result}', 期望: '{expected}'")
            all_correct = False

    return all_correct


# ==============================================================================
# 练习 2：实现倒排索引
# ==============================================================================

def exercise_2_inverted_index(documents: list) -> dict:
    """
    练习 2：构建倒排索引

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你在编一本书的"索引"：
    - 翻到第 1 页，看到"机器学习"，记下来：机器学习 → [第 1 页]
    - 翻到第 2 页，看到"深度学习"，记下来：深度学习 → [第 2 页]
    - 翻到第 3 页，又看到"机器学习"，记下来：机器学习 → [第 1, 3 页]

    ━━━━━━━ 提示 ━━━━━━━
    1. 创建一个字典 index = defaultdict(list)

    2. 遍历每篇文档（用 enumerate 获取文档 ID）：
       a. 对文档进行分词（按字符分词，跳过空白和标点）
       b. 统计每个词在这篇文档中出现的次数
       c. 把 (文档ID, 词频) 添加到索引中

    3. 返回索引字典

    参数：
        documents: 文档列表（字符串列表）

    返回：
        倒排索引 {词: [(文档ID, 词频), ...]}

    示例：
        >>> docs = ["机器学习", "深度学习"]
        >>> exercise_2_inverted_index(docs)
        {"机": [(0, 1)], "器": [(0, 1)], ...}
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # index = defaultdict(list)
    # for doc_id, doc in enumerate(documents):
    #     # 分词
    #     tokens = []
    #     for char in doc:
    #         if char.strip() and char not in "，。！？、；：""''（）【】《》\n\r\t":
    #             tokens.append(char)
    #     # 统计词频
    #     term_freq = Counter(tokens)
    #     # 更新索引
    #     for term, freq in term_freq.items():
    #         index[term].append((doc_id, freq))
    # return dict(index)
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：构建倒排索引")
    print("=" * 60)

    documents = ["机器学习", "深度学习"]

    result = exercise_2_inverted_index(documents)

    if result is None:
        print("[未完成] 请实现 exercise_2_inverted_index 函数")
        return False

    if not isinstance(result, dict):
        print(f"[错误] 返回值应该是字典")
        return False

    # 检查关键字符是否在索引中
    expected_chars = ["机", "学", "深"]
    all_found = True
    for char in expected_chars:
        if char in result:
            print(f"  [正确] '{char}' 在索引中: {result[char]}")
        else:
            print(f"  [错误] '{char}' 不在索引中")
            all_found = False

    return all_found


# ==============================================================================
# 练习 3：实现简单的协同过滤推荐
# ==============================================================================

def exercise_3_recommend(ratings: dict, user: str, top_k: int = 2) -> list:
    """
    练习 3：实现简单的基于用户的协同过滤推荐

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你想找一部好电影看：
    1. 找到和你口味最相似的朋友
    2. 看看他们喜欢什么你没看过的
    3. 推荐给你

    ━━━━━━━ 提示 ━━━━━━━
    1. 计算目标用户与其他用户的余弦相似度
       - 找到两个用户共同评分的物品
       - 余弦相似度 = Σ(r1i * r2i) / (sqrt(Σr1i^2) * sqrt(Σr2i^2))

    2. 找到目标用户未评分的物品

    3. 对每个未评分的物品，计算预测评分
       - 预测评分 = Σ(相似度 * 其他用户的评分) / Σ(|相似度|)

    4. 按预测评分排序，返回前 top_k 个

    参数：
        ratings: 评分字典 {用户: {物品: 评分}}
        user: 目标用户
        top_k: 返回前 k 个推荐

    返回：
        推荐列表，每个元素是 (物品ID, 预测评分)

    示例：
        >>> ratings = {
        ...     "小明": {"电影A": 5, "电影B": 4},
        ...     "小红": {"电影A": 5, "电影C": 4},
        ... }
        >>> exercise_3_recommend(ratings, "小明")
        [("电影C", 预测评分)]
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # # 获取目标用户的评分
    # user_ratings = ratings[user]
    # user_items = set(user_ratings.keys())
    #
    # # 计算与其他用户的相似度
    # similarities = []
    # for other_user, other_ratings in ratings.items():
    #     if other_user == user:
    #         continue
    #
    #     # 共同评分的物品
    #     common_items = user_items & set(other_ratings.keys())
    #     if not common_items:
    #         continue
    #
    #     # 余弦相似度
    #     dot = sum(user_ratings[i] * other_ratings[i] for i in common_items)
    #     norm1 = math.sqrt(sum(user_ratings[i] ** 2 for i in common_items))
    #     norm2 = math.sqrt(sum(other_ratings[i] ** 2 for i in common_items))
    #
    #     if norm1 > 0 and norm2 > 0:
    #         sim = dot / (norm1 * norm2)
    #         if sim > 0:
    #             similarities.append((other_user, sim))
    #
    # # 按相似度排序
    # similarities.sort(key=lambda x: x[1], reverse=True)
    #
    # # 获取候选物品（目标用户未评分的）
    # candidate_items = set()
    # for other_user, _ in similarities:
    #     candidate_items |= set(ratings[other_user].keys())
    # candidate_items -= user_items
    #
    # # 计算预测评分
    # predictions = []
    # for item in candidate_items:
    #     numerator = 0
    #     denominator = 0
    #     for other_user, sim in similarities:
    #         if item in ratings[other_user]:
    #             numerator += sim * ratings[other_user][item]
    #             denominator += abs(sim)
    #     if denominator > 0:
    #         predictions.append((item, numerator / denominator))
    #
    # predictions.sort(key=lambda x: x[1], reverse=True)
    # return predictions[:top_k]
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：协同过滤推荐")
    print("=" * 60)

    ratings = {
        "小明": {"肖申克": 5, "阿甘正传": 4},
        "小红": {"肖申克": 5, "阿甘正传": 4, "盗梦空间": 4},
        "小刚": {"盗梦空间": 5, "星际穿越": 5},
    }

    result = exercise_3_recommend(ratings, "小明", top_k=2)

    if result is None:
        print("[未完成] 请实现 exercise_3_recommend 函数")
        return False

    if not isinstance(result, list) or len(result) == 0:
        print(f"[错误] 返回值应该是非空列表")
        return False

    if not isinstance(result[0], tuple) or len(result[0]) != 2:
        print(f"[错误] 返回值应该是 (物品, 评分) 元组列表")
        return False

    # 检查推荐的物品是否是小明没看过的
    user_items = set(ratings["小明"].keys())
    recommended_items = [item for item, _ in result]

    all_new = all(item not in user_items for item in recommended_items)

    if all_new:
        print(f"[正确] 为小明推荐:")
        for item, score in result:
            print(f"       {item}: 预测评分 {score:.2f}")
        return True
    else:
        print(f"[错误] 推荐了用户已经看过的物品")
        return False


# ==============================================================================
# 主程序：运行所有练习测试
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十九章 练习                  ║
    ║        项目实战                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 运行所有练习测试
    results = []
    results.append(("练习1: 聊天机器人规则匹配", test_exercise_1()))
    results.append(("练习2: 倒排索引", test_exercise_2()))
    results.append(("练习3: 协同过滤推荐", test_exercise_3()))

    # 练习清单
    print("\n" + "=" * 60)
    print("  练习清单")
    print("=" * 60)
    for name, passed in results:
        status = "[完成]" if passed else "[未完成]"
        print(f"  {status} {name}")

    # 计算完成率
    completed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  完成率: {completed}/{total}")

    if completed == total:
        print("\n  恭喜！所有练习都完成了！")
        print("  你已经掌握了 NLP 项目实战的核心技能。")
        print("  继续加油，NLP 的世界还有更多精彩等着你！")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
        print("  不要着急，慢慢来，理解了再写代码。")
