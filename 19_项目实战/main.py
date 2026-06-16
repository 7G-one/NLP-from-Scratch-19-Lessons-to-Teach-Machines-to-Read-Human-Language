import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第十九章：项目实战 — 完整演示
==============================================================================
G-one NLP 学院
日期：2026-05-16

运行方式：
    python main.py

前置知识：
    - 前面所有章节的内容

本章内容：
    1. 聊天机器人（基于规则的对话系统）
    2. 搜索引擎（完整的搜索系统）
    3. 推荐系统（协同过滤 + 内容推荐）
==============================================================================
"""

from chatbot import ChatBot
from search_project import SearchEngine
from recommend import CollaborativeFiltering, ContentBasedRecommender, HybridRecommender


def print_separator(title: str):
    """打印分隔线和标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_chatbot():
    """演示聊天机器人"""

    print_separator("19.1 聊天机器人演示")

    print("""
    ┌─────────────────────────────────────────────────────────┐
    │              聊天机器人 = 自动问答系统                    │
    │                                                         │
    │   核心组件：                                             │
    │   1. 模式匹配：识别用户输入中的关键词                    │
    │   2. 意图识别：判断用户想做什么                          │
    │   3. 实体提取：提取关键信息                              │
    │   4. 对话管理：维护上下文                                │
    └─────────────────────────────────────────────────────────┘
    """)

    bot = ChatBot(name="小智")

    test_inputs = [
        "你好呀",
        "今天天气怎么样？",
        "你能做什么？",
        "谢谢",
        "再见",
    ]

    print("  对话演示:")
    for user_input in test_inputs:
        response = bot.chat(user_input)
        intent = bot.get_intent(user_input)
        print(f"\n    用户: {user_input}")
        print(f"    意图: {intent}")
        print(f"    机器人: {response}")


def demo_search_engine():
    """演示搜索引擎"""

    print_separator("19.2 搜索引擎演示")

    print("""
    ┌─────────────────────────────────────────────────────────┐
    │              搜索引擎 = 智能检索系统                      │
    │                                                         │
    │   核心组件：                                             │
    │   1. 倒排索引：词 → 文档的映射                          │
    │   2. TF-IDF 排序：基于词频和文档频率                    │
    │   3. BM25 排序：更先进的排序算法                        │
    │   4. 查询处理：解析用户查询                             │
    └─────────────────────────────────────────────────────────┘
    """)

    engine = SearchEngine()

    documents = {
        0: "机器学习是人工智能的重要分支。它让计算机能够从数据中学习。",
        1: "深度学习是机器学习的热门方向。它使用多层神经网络。",
        2: "自然语言处理让计算机理解人类语言。分词是基础任务。",
        3: "卷积神经网络在图像识别领域取得了巨大成功。",
        4: "循环神经网络适合处理序列数据，如文本和语音。",
        5: "今天股市大涨，上证指数突破三千点。",
        6: "央行宣布降息，刺激经济增长。",
    }

    print("  文档库:")
    for doc_id, text in documents.items():
        engine.add_document(doc_id, text)
        print(f"    [{doc_id}] {text[:30]}...")

    queries = ["机器学习", "神经网络", "股市"]

    print("\n  搜索结果:")
    for query in queries:
        results = engine.search(query, top_k=2)
        print(f"\n    查询: '{query}'")
        for doc_id, score, text in results:
            print(f"      [{doc_id}] {score:.4f} - {text[:40]}...")


def demo_recommendation():
    """演示推荐系统"""

    print_separator("19.3 推荐系统演示")

    print("""
    ┌─────────────────────────────────────────────────────────┐
    │              推荐系统 = 智能推荐引擎                      │
    │                                                         │
    │   两种主要方法：                                         │
    │   ┌─────────────┬──────────────────────────────────┐    │
    │   │ 协同过滤    │ 基于用户行为的相似性              │    │
    │   │ 内容推荐    │ 基于物品特征的相似性              │    │
    │   └─────────────┴──────────────────────────────────┘    │
    │                                                         │
    │   混合推荐：结合两种方法的优点                          │
    └─────────────────────────────────────────────────────────┘
    """)

    # 协同过滤
    print("  [方法 1] 协同过滤:")
    cf = CollaborativeFiltering()

    ratings = [
        ("小明", "肖申克的救赎", 5),
        ("小明", "阿甘正传", 4),
        ("小明", "泰坦尼克号", 5),
        ("小红", "肖申克的救赎", 5),
        ("小红", "阿甘正传", 4),
        ("小红", "盗梦空间", 4),
        ("小刚", "泰坦尼克号", 5),
        ("小刚", "盗梦空间", 5),
        ("小刚", "星际穿越", 5),
    ]

    for user, item, rating in ratings:
        cf.add_rating(user, item, rating)

    recs = cf.recommend_user_based("小明", top_k=3)
    print("    为小明推荐:")
    for item, score in recs:
        print(f"      {item}: {score:.2f}")

    # 内容推荐
    print("\n  [方法 2] 内容推荐:")
    cb = ContentBasedRecommender()

    cb.add_item("肖申克的救赎", {"剧情": 5, "动作": 1, "科幻": 0})
    cb.add_item("阿甘正传", {"剧情": 5, "动作": 1, "科幻": 0})
    cb.add_item("盗梦空间", {"剧情": 4, "动作": 4, "科幻": 5})
    cb.add_item("星际穿越", {"剧情": 4, "动作": 2, "科幻": 5})

    cb.add_rating("小明", "肖申克的救赎", 5)
    cb.add_rating("小明", "阿甘正传", 4)

    recs = cb.recommend("小明", top_k=3)
    print("    为小明推荐:")
    for item, score in recs:
        print(f"      {item}: {score:.2f}")


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十九章                      ║
    ║        项目实战                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    demo_chatbot()
    demo_search_engine()
    demo_recommendation()

    # 课程总结
    print("\n" + "=" * 60)
    print("  第十九章 总结")
    print("=" * 60)
    print("""
    [OK] 聊天机器人 — 基于规则的对话系统
    [OK] 搜索引擎 — 倒排索引 + TF-IDF/BM25 排序
    [OK] 推荐系统 — 协同过滤 + 内容推荐 + 混合推荐
    """)

    print("=" * 60)
    print("  NLP 课程全部完成！")
    print("=" * 60)
    print("""
    恭喜你完成了 G-one NLP 学院的全部课程！

    你已经掌握了：
    - NLP 基础：分词、词性标注、命名实体识别
    - 文本分析：相似度、聚类、关键词提取、摘要
    - 语言模型：N-Gram、困惑度、LSTM
    - 深度学习：神经网络、CNN、RNN、LSTM
    - 项目实战：聊天机器人、搜索引擎、推荐系统

    下一步建议：
    1. 多做项目，把学到的知识应用到实际场景
    2. 学习更先进的技术（BERT、GPT、Transformer）
    3. 参加 NLP 竞赛，提升实战能力
    4. 阅读论文，了解最新研究进展

    祝你在 NLP 的道路上越走越远！
    """)
