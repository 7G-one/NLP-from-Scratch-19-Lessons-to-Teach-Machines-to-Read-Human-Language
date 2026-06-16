"""
==============================================================================
第十九章：项目实战 — 推荐系统
==============================================================================
日期：2026-05-16

同学们好！这个模块我们来实现一个"推荐系统"。

----------------------------------------------------------------------
生活类比：推荐系统就像"朋友推荐"
----------------------------------------------------------------------

想象你想看一部电影，你会怎么做？

  方法 1：问和你口味相似的朋友（协同过滤）
    - 小明和你都喜欢《肖申克的救赎》《阿甘正传》
    - 小明还推荐了《泰坦尼克号》
    - 你可能也会喜欢《泰坦尼克号》

  方法 2：找和你喜欢的电影相似的电影（内容推荐）
    - 你喜欢《肖申克的救赎》（剧情片、高评分）
    - 找其他"剧情片、高评分"的电影
    - 推荐给你

这就是推荐系统的两种主要方法：
  - 协同过滤：基于用户行为的相似性
  - 内容推荐：基于物品特征的相似性

----------------------------------------------------------------------
本模块内容
----------------------------------------------------------------------

1. 协同过滤（Collaborative Filtering）
   - 用户-based 协同过滤
   - 物品-based 协同过滤
2. 内容推荐（Content-Based）
3. 评估指标

==============================================================================
"""

import math
from collections import defaultdict


# ==============================================================================
# 第一部分：协同过滤（Collaborative Filtering）
# ==============================================================================
#
# 协同过滤是最经典的推荐算法。
#
# 核心思想：
#   找到和你"口味相似"的人，把他们喜欢的东西推荐给你。
#
# 生活类比：
#   想象你是一个美食家：
#   - 你和小明都喜欢火锅、烧烤
#   - 小明还推荐了一家日料店
#   - 你可能也会喜欢那家日料店
#
# ==============================================================================


class CollaborativeFiltering:
    """
    协同过滤推荐系统

    ━━━━━━━ 生活类比 ━━━━━━━
    就像"物以类聚，人以群分"：
    - 找到和你口味相似的人
    - 把他们喜欢的推荐给你
    """

    def __init__(self):
        """初始化协同过滤"""
        # 用户-物品评分矩阵
        # {用户ID: {物品ID: 评分}}
        self.ratings = defaultdict(dict)
        # 所有用户和物品
        self.users = set()
        self.items = set()

    def add_rating(self, user: str, item: str, rating: float):
        """
        添加用户评分

        参数：
            user: 用户 ID
            item: 物品 ID
            rating: 评分（1-5）
        """
        self.ratings[user][item] = rating
        self.users.add(user)
        self.items.add(item)

    def cosine_similarity(self, user1: str, user2: str) -> float:
        """
        计算两个用户的相似度（余弦相似度）

        ━━━━━━━ 生活类比 ━━━━━━━
        就像比较两个人的"口味偏好"：
        - 如果两个人对同一批餐厅的评分很接近 → 相似度高
        - 如果评分差异很大 → 相似度低

        公式：
            sim(u1, u2) = Σ(r1i * r2i) / (||r1|| * ||r2||)
        """
        # 获取两个用户共同评分的物品
        common_items = set(self.ratings[user1].keys()) & set(self.ratings[user2].keys())

        if not common_items:
            return 0.0

        # 计算余弦相似度
        dot_product = sum(self.ratings[user1][item] * self.ratings[user2][item]
                         for item in common_items)

        norm1 = math.sqrt(sum(r ** 2 for r in self.ratings[user1].values()))
        norm2 = math.sqrt(sum(r ** 2 for r in self.ratings[user2].values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def pearson_similarity(self, user1: str, user2: str) -> float:
        """
        计算两个用户的皮尔逊相关系数

        ━━━━━━━ 生活类比 ━━━━━━━
        皮尔逊相关系数比余弦相似度更"公平"：
        - 如果一个人打分普遍偏高，另一个人普遍偏低
        - 但他们的"相对偏好"是一样的
        - 皮尔逊系数能识别这种情况

        公式：
            sim(u1, u2) = Σ((r1i - μ1) * (r2i - μ2)) / (σ1 * σ2)
        """
        common_items = set(self.ratings[user1].keys()) & set(self.ratings[user2].keys())

        if len(common_items) < 2:
            return 0.0

        # 计算平均评分
        mean1 = sum(self.ratings[user1][item] for item in common_items) / len(common_items)
        mean2 = sum(self.ratings[user2][item] for item in common_items) / len(common_items)

        # 计算皮尔逊相关系数
        numerator = sum((self.ratings[user1][item] - mean1) * (self.ratings[user2][item] - mean2)
                        for item in common_items)

        denom1 = math.sqrt(sum((self.ratings[user1][item] - mean1) ** 2 for item in common_items))
        denom2 = math.sqrt(sum((self.ratings[user2][item] - mean2) ** 2 for item in common_items))

        if denom1 == 0 or denom2 == 0:
            return 0.0

        return numerator / (denom1 * denom2)

    def recommend_user_based(self, user: str, top_k: int = 3) -> list:
        """
        基于用户的协同过滤推荐

        ━━━━━━━ 生活类比 ━━━━━━━
        就像问"和你口味相似的朋友"：
        1. 找到和你最相似的 K 个用户
        2. 看看他们喜欢什么你没看过的
        3. 按相似度加权推荐

        参数：
            user: 目标用户
            top_k: 返回前 k 个推荐

        返回：
            推荐列表，每个元素是 (物品ID, 预测评分)
        """
        if user not in self.users:
            return []

        # 计算目标用户与其他用户的相似度
        similarities = []
        for other_user in self.users:
            if other_user != user:
                sim = self.pearson_similarity(user, other_user)
                if sim > 0:
                    similarities.append((other_user, sim))

        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 获取目标用户未评分的物品
        user_items = set(self.ratings[user].keys())
        candidate_items = self.items - user_items

        # 对每个候选物品计算预测评分
        predictions = []
        for item in candidate_items:
            numerator = 0
            denominator = 0

            for other_user, sim in similarities:
                if item in self.ratings[other_user]:
                    numerator += sim * self.ratings[other_user][item]
                    denominator += abs(sim)

            if denominator > 0:
                predicted_rating = numerator / denominator
                predictions.append((item, predicted_rating))

        # 按预测评分降序排列
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_k]

    def recommend_item_based(self, user: str, top_k: int = 3) -> list:
        """
        基于物品的协同过滤推荐

        ━━━━━━━ 生活类比 ━━━━━━━
        就像"找相似的电影"：
        1. 找到用户喜欢的物品
        2. 找到和这些物品最相似的其他物品
        3. 推荐相似物品

        参数：
            user: 目标用户
            top_k: 返回前 k 个推荐

        返回：
            推荐列表，每个元素是 (物品ID, 预测评分)
        """
        if user not in self.users:
            return []

        user_items = set(self.ratings[user].keys())
        candidate_items = self.items - user_items

        # 计算物品之间的相似度
        predictions = []
        for candidate in candidate_items:
            numerator = 0
            denominator = 0

            for rated_item in user_items:
                # 计算物品相似度
                sim = self._item_similarity(candidate, rated_item)
                if sim > 0:
                    numerator += sim * self.ratings[user][rated_item]
                    denominator += abs(sim)

            if denominator > 0:
                predicted_rating = numerator / denominator
                predictions.append((candidate, predicted_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def _item_similarity(self, item1: str, item2: str) -> float:
        """
        计算两个物品的相似度

        思路：如果两个物品被同一批用户相似地评分，则它们相似
        """
        # 获取对这两个物品都评过分的用户
        common_users = [u for u in self.users
                        if item1 in self.ratings[u] and item2 in self.ratings[u]]

        if len(common_users) < 2:
            return 0.0

        # 计算余弦相似度
        dot_product = sum(self.ratings[u][item1] * self.ratings[u][item2] for u in common_users)
        norm1 = math.sqrt(sum(self.ratings[u][item1] ** 2 for u in common_users))
        norm2 = math.sqrt(sum(self.ratings[u][item2] ** 2 for u in common_users))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


# ==============================================================================
# 第二部分：内容推荐（Content-Based）
# ==============================================================================
#
# 内容推荐的核心思想：
#   根据物品的"特征"来推荐相似的物品。
#
# 生活类比：
#   想象你喜欢吃"辣的、四川菜"：
#   - 系统找到其他"辣的、四川菜"餐厅
#   - 推荐给你
#
# ==============================================================================


class ContentBasedRecommender:
    """
    基于内容的推荐系统

    ━━━━━━━ 生活类比 ━━━━━━━
    就像"找相似的东西"：
    - 你喜欢红色的苹果
    - 系统推荐其他红色的水果（草莓、樱桃）
    """

    def __init__(self):
        """初始化内容推荐"""
        # 物品特征
        # {物品ID: {特征1: 值, 特征2: 值, ...}}
        self.item_features = defaultdict(dict)
        # 用户偏好
        # {用户ID: {特征1: 权重, 特征2: 权重, ...}}
        self.user_preferences = defaultdict(lambda: defaultdict(float))
        # 用户评分
        self.ratings = defaultdict(dict)

    def add_item(self, item: str, features: dict):
        """
        添加物品及其特征

        参数：
            item: 物品 ID
            features: 特征字典 {特征名: 特征值}
        """
        self.item_features[item] = features

    def add_rating(self, user: str, item: str, rating: float):
        """
        添加用户评分（用于学习用户偏好）

        参数：
            user: 用户 ID
            item: 物品 ID
            rating: 评分
        """
        self.ratings[user][item] = rating

        # 更新用户偏好（加权平均）
        if item in self.item_features:
            for feature, value in self.item_features[item].items():
                # 用评分加权特征值
                self.user_preferences[user][feature] += rating * value

    def _normalize_preferences(self, user: str):
        """归一化用户偏好"""
        total = sum(abs(v) for v in self.user_preferences[user].values())
        if total > 0:
            for feature in self.user_preferences[user]:
                self.user_preferences[user][feature] /= total

    def recommend(self, user: str, top_k: int = 3) -> list:
        """
        基于内容推荐

        ━━━━━━━ 流程 ━━━━━━━
        1. 分析用户历史评分，学习用户偏好
        2. 计算每个候选物品与用户偏好的匹配度
        3. 按匹配度排序，返回 top_k 个

        参数：
            user: 目标用户
            top_k: 返回前 k 个推荐

        返回：
            推荐列表，每个元素是 (物品ID, 匹配度)
        """
        # 归一化用户偏好
        self._normalize_preferences(user)

        # 获取用户未评分的物品
        user_items = set(self.ratings[user].keys())
        candidate_items = set(self.item_features.keys()) - user_items

        # 计算每个候选物品的匹配度
        scores = []
        for item in candidate_items:
            score = 0
            for feature, value in self.item_features[item].items():
                # 特征值 × 用户对这个特征的偏好
                score += value * self.user_preferences[user].get(feature, 0)
            scores.append((item, score))

        # 按匹配度降序排列
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


# ==============================================================================
# 第三部分：混合推荐
# ==============================================================================


class HybridRecommender:
    """
    混合推荐系统

    ━━━━━━━ 生活类比 ━━━━━━━
    就像"综合多方意见"：
    - 问朋友推荐（协同过滤）
    - 自己看特征匹配（内容推荐）
    - 综合两方面意见做决定
    """

    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.4):
        """
        初始化混合推荐

        参数：
            cf_weight: 协同过滤的权重
            cb_weight: 内容推荐的权重
        """
        self.cf = CollaborativeFiltering()
        self.cb = ContentBasedRecommender()
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight

    def add_item(self, item: str, features: dict):
        """添加物品"""
        self.cb.add_item(item, features)

    def add_rating(self, user: str, item: str, rating: float):
        """添加评分"""
        self.cf.add_rating(user, item, rating)
        self.cb.add_rating(user, item, rating)

    def recommend(self, user: str, top_k: int = 3) -> list:
        """
        混合推荐

        参数：
            user: 目标用户
            top_k: 返回前 k 个推荐

        返回：
            推荐列表
        """
        # 协同过滤推荐
        cf_recs = self.cf.recommend_user_based(user, top_k=top_k * 2)
        cf_dict = {item: score for item, score in cf_recs}

        # 内容推荐
        cb_recs = self.cb.recommend(user, top_k=top_k * 2)
        cb_dict = {item: score for item, score in cb_recs}

        # 合并分数
        all_items = set(cf_dict.keys()) | set(cb_dict.keys())
        combined = []
        for item in all_items:
            cf_score = cf_dict.get(item, 0)
            cb_score = cb_dict.get(item, 0)
            total_score = self.cf_weight * cf_score + self.cb_weight * cb_score
            combined.append((item, total_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十九章                      ║
    ║        项目实战：推荐系统                             ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # ━━━━━━━ 协同过滤演示 ━━━━━━━
    print("=" * 60)
    print("  1. 协同过滤推荐")
    print("=" * 60)

    cf = CollaborativeFiltering()

    # 添加评分数据
    ratings_data = [
        ("小明", "肖申克的救赎", 5),
        ("小明", "阿甘正传", 4),
        ("小明", "泰坦尼克号", 5),
        ("小红", "肖申克的救赎", 5),
        ("小红", "阿甘正传", 4),
        ("小红", "盗梦空间", 4),
        ("小刚", "泰坦尼克号", 5),
        ("小刚", "盗梦空间", 5),
        ("小刚", "星际穿越", 5),
        ("小李", "肖申克的救赎", 4),
        ("小李", "星际穿越", 5),
        ("小李", "盗梦空间", 4),
    ]

    for user, item, rating in ratings_data:
        cf.add_rating(user, item, rating)

    # 为小明推荐
    print("\n  为小明推荐（基于用户）:")
    recs = cf.recommend_user_based("小明", top_k=3)
    for item, score in recs:
        print(f"    {item}: 预测评分 {score:.2f}")

    print("\n  为小明推荐（基于物品）:")
    recs = cf.recommend_item_based("小明", top_k=3)
    for item, score in recs:
        print(f"    {item}: 预测评分 {score:.2f}")

    # ━━━━━━━ 内容推荐演示 ━━━━━━━
    print("\n" + "=" * 60)
    print("  2. 内容推荐")
    print("=" * 60)

    cb = ContentBasedRecommender()

    # 添加物品特征
    cb.add_item("肖申克的救赎", {"剧情": 5, "动作": 1, "科幻": 0, "喜剧": 1})
    cb.add_item("阿甘正传", {"剧情": 5, "动作": 1, "科幻": 0, "喜剧": 3})
    cb.add_item("泰坦尼克号", {"剧情": 5, "动作": 2, "科幻": 0, "喜剧": 1})
    cb.add_item("盗梦空间", {"剧情": 4, "动作": 4, "科幻": 5, "喜剧": 1})
    cb.add_item("星际穿越", {"剧情": 4, "动作": 2, "科幻": 5, "喜剧": 1})
    cb.add_item("功夫", {"剧情": 2, "动作": 4, "科幻": 0, "喜剧": 5})

    # 添加评分
    cb.add_rating("小明", "肖申克的救赎", 5)
    cb.add_rating("小明", "阿甘正传", 4)

    print("\n  为小明推荐（基于内容）:")
    recs = cb.recommend("小明", top_k=3)
    for item, score in recs:
        print(f"    {item}: 匹配度 {score:.2f}")

    # ━━━━━━━ 混合推荐演示 ━━━━━━━
    print("\n" + "=" * 60)
    print("  3. 混合推荐")
    print("=" * 60)

    hybrid = HybridRecommender(cf_weight=0.6, cb_weight=0.4)

    # 添加物品特征
    hybrid.add_item("肖申克的救赎", {"剧情": 5, "动作": 1, "科幻": 0})
    hybrid.add_item("阿甘正传", {"剧情": 5, "动作": 1, "科幻": 0})
    hybrid.add_item("泰坦尼克号", {"剧情": 5, "动作": 2, "科幻": 0})
    hybrid.add_item("盗梦空间", {"剧情": 4, "动作": 4, "科幻": 5})
    hybrid.add_item("星际穿越", {"剧情": 4, "动作": 2, "科幻": 5})

    # 添加评分
    for user, item, rating in ratings_data:
        hybrid.add_rating(user, item, rating)

    print("\n  为小明推荐（混合）:")
    recs = hybrid.recommend("小明", top_k=3)
    for item, score in recs:
        print(f"    {item}: 综合得分 {score:.2f}")
