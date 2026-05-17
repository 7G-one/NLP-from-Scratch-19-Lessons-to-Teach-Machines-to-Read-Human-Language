import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第二章：中文分词 — 完整演示
==============================================================================
G-one NLP 学院
日期：2026-05-16

运行方式：
    python main.py

前置知识：
    - 第一章：NLP 概述

本章内容：
    1. 规则分词：正向最大匹配、逆向最大匹配、双向最大匹配
    2. 统计分词：HMM 分词模型
    3. jieba 分词实战
    4. 分词效果评估
==============================================================================
"""

from segmentation import (
    forward_max_match,
    backward_max_match,
    bidirectional_max_match,
    HMMSegmentation,
    DEFAULT_DICT,
    evaluate_segmentation,
)


def print_separator(title: str):
    """打印分隔线和标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def lesson_rule_based():
    """第一部分：规则分词"""

    print_separator("2.1 规则分词 — 基于词典的分词")

    print("""
    规则分词是最简单的分词方法：
    1. 准备一个词典（包含所有已知的词）
    2. 按照规则，从句子中匹配词典里的词

    三种方法：
    - 正向最大匹配（FMM）：从左到右，每次切最大的词
    - 逆向最大匹配（BMM）：从右到左，每次切最大的词
    - 双向最大匹配：两种都试，选更好的
    """)

    # 演示 1：简单句子
    text = "今天天气真好"
    dictionary = {"今天", "天气", "真好"}
    print(f"\n示例 1: '{text}'")
    print(f"词典: {dictionary}")
    print(f"  FMM: {forward_max_match(text, dictionary)}")
    print(f"  BMM: {backward_max_match(text, dictionary)}")
    print(f"  双向: {bidirectional_max_match(text, dictionary)}")

    # 演示 2：有歧义的句子
    text = "南京市有长江大桥"
    dictionary = {"南京市", "有", "长江大桥", "南京", "长江", "大桥", "市长"}
    print(f"\n示例 2: '{text}'")
    print(f"词典: {dictionary}")
    fmm = forward_max_match(text, dictionary)
    bmm = backward_max_match(text, dictionary)
    bi = bidirectional_max_match(text, dictionary)
    print(f"  FMM: {fmm}")
    print(f"  BMM: {bmm}")
    print(f"  双向: {bi}")

    # 演示 3：歧义分析
    print("\n" + "-" * 40)
    print("歧义分析:")
    print("  '南京市有长江大桥'")
    print("  FMM 切法: 南京市 / 有 / 长江大桥  ← 正确！")
    print("  BMM 切法: 南京 / 市 / 有 / 长江大桥  ← 也可以")
    print("  两种方法结果不同，双向最大匹配会选择更好的那个")


def lesson_hmm():
    """第二部分：HMM 统计分词"""

    print_separator("2.2 HMM 统计分词")

    print("""
    HMM（隐马尔可夫模型）是一种统计分词方法。

    核心思想：
    - 给每个字标记一个状态：B(词首)、M(词中)、B(词尾)、S(单字)
    - 例如："今天天气" → B E B E
    - 根据状态序列确定分词边界

    就像侦探破案：根据线索推断每个字的"身份"。
    """)

    # 创建并训练 HMM 模型
    hmm = HMMSegmentation()

    training_data = [
        (['今', '天', '天', '气', '真', '好'], ['B', 'E', 'B', 'E', 'S', 'S']),
        (['我', '喜', '欢', '学', '习'], ['S', 'B', 'E', 'B', 'E']),
        (['自', '然', '语', '言', '处', '理'], ['B', 'E', 'B', 'E', 'B', 'E']),
        (['深', '度', '学', '习', '很', '有', '趣'], ['B', 'E', 'B', 'E', 'S', 'B', 'E']),
        (['机', '器', '学', '习', '是', '人', '工', '智', '能', '的', '一', '个', '分', '支'],
         ['B', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'S', 'S', 'B', 'E']),
    ]

    hmm.train(training_data)
    print("HMM 模型训练完成！")

    test_texts = [
        "今天天气好",
        "我喜欢学习",
        "深度学习有趣",
        "机器学习是人工智能",
    ]

    for text in test_texts:
        result = hmm.segment(text)
        print(f"  '{text}' → {' / '.join(result)}")


def lesson_jieba():
    """第三部分：jieba 分词实战"""

    print_separator("2.3 jieba 分词实战")

    try:
        import jieba
        import jieba.posseg as pseg
    except ImportError:
        print("[提示] jieba 未安装，请运行: pip install jieba")
        return

    test_text = "我来到北京清华大学学习自然语言处理"
    print(f"\n测试文本: {test_text}")

    # 精确模式
    print("\n--- 精确模式（最常用）---")
    words = jieba.cut(test_text, cut_all=False)
    print(f"结果: {' / '.join(words)}")

    # 全模式
    print("\n--- 全模式（所有可能的词）---")
    words = jieba.cut(test_text, cut_all=True)
    print(f"结果: {' / '.join(words)}")

    # 搜索引擎模式
    print("\n--- 搜索引擎模式 ---")
    words = jieba.cut_for_search(test_text)
    print(f"结果: {' / '.join(words)}")

    # 词性标注
    print("\n--- 词性标注 ---")
    words_pos = pseg.cut(test_text)
    for word, flag in words_pos:
        print(f"  {word} → {flag}")

    # 自定义词典
    print("\n--- 自定义词典 ---")
    jieba.add_word("赵本山", freq=10000, tag="nr")
    jieba.add_word("上春晚", freq=10000, tag="v")
    print("添加自定义词后:")
    print("  '赵本山今年上春晚' →", " / ".join(jieba.cut("赵本山今年上春晚")))


def lesson_evaluation():
    """第四部分：分词评估"""

    print_separator("2.4 分词效果评估")

    print("""
    评估分词效果的三个指标：

    准确率（Precision）= 正确的词数 / 切出来的总词数
    召回率（Recall）   = 正确的词数 / 标准答案中的总词数
    F1 值             = 2 × P × R / (P + R)
    """)

    # 模拟评估
    golden = ["今天", "天气", "真好"]
    predicted_good = ["今天", "天气", "真好"]
    predicted_bad = ["今", "天天", "气", "真好"]

    print(f"标准答案: {golden}")
    print(f"\n切分结果 1 (好的): {predicted_good}")
    result1 = evaluate_segmentation(predicted_good, golden)
    print(f"  准确率: {result1['precision']}")
    print(f"  召回率: {result1['recall']}")
    print(f"  F1: {result1['f1']}")

    print(f"\n切分结果 2 (差的): {predicted_bad}")
    result2 = evaluate_segmentation(predicted_bad, golden)
    print(f"  准确率: {result2['precision']}")
    print(f"  召回率: {result2['recall']}")
    print(f"  F1: {result2['f1']}")


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第二章                        ║
    ║        中文分词                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    lesson_rule_based()
    lesson_hmm()
    lesson_jieba()
    lesson_evaluation()

    # 课程总结
    print("\n" + "=" * 60)
    print("  第二章 总结")
    print("=" * 60)
    print("""
    [OK] 规则分词 — 基于词典，简单快速
         FMM（正向） → 从左到右切
         BMM（逆向） → 从右到左切
         双向最大匹配 → 取两者中更好的

    [OK] HMM 统计分词 — 基于概率
         B/M/E/S 标记法
         维特比算法找最优路径

    [OK] jieba 分词实战
         精确模式 / 全模式 / 搜索引擎模式
         自定义词典

    [OK] 分词评估
         准确率、召回率、F1 值
    """)

    print("-" * 60)
    print("  下节预告：第三章 — 词性标注")
    print("-" * 60)
    print("""
    下一章我们将学习：
    - 什么是词性（名词、动词、形容词等）
    - HMM 词性标注原理
    - jieba 词性标注实战
    """)
