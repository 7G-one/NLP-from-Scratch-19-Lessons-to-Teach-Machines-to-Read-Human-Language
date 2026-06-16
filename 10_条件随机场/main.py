import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    import numpy as np
except ImportError:
    print("本课需要 numpy 库，请运行：pip install numpy")
    print("安装后重新运行本文件即可。")
    exit(1)

"""
==============================================================================
第十章：条件随机场（CRF）— 完整演示
==============================================================================
G-one NLP 学院
日期：2026-05-16

运行方式：
    python main.py

前置知识：
    - 第三章：词性标注
    - 第四章：命名实体识别

本章内容：
    1. CRF 基本概念
    2. 特征模板设计
    3. 简化 CRF 实现
    4. sklearn-crfsuite 实战
==============================================================================
"""

from crf import (
    softmax,
    word_features,
    word_features_chinese,
    extract_sentence_features,
    SimpleCRF,
    demo_sklearn_crfsuite,
    demo_crf_feature_importance,
    explain_crf_vs_hmm,
    visualize_crf_transition,
)


def print_separator(title: str):
    """打印分隔线和标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def lesson_crf_concept():
    """第一部分：CRF 基本概念"""

    print_separator("10.1 CRF 基本概念")

    print("""
    条件随机场（Conditional Random Field, CRF）是一种判别式模型。

    核心思想：
      不是孤立地看每个词，而是综合考虑"前后文"来判断每个词的标签。

    生活类比：
      想象你是一个侦探，面前有一排人，你需要判断每个人的职业。

      如果你只看一个人的外表，很难判断。
      但如果你看"上下文"，就容易多了：

        ... 医生 护士 [?] 病人 ...
        → 问号处很可能也是医院里的人

        ... 厨师 服务员 [?] 顾客 ...
        → 问号处很可能是餐厅里的人

    CRF 的优势：
      1. 不需要独立性假设（比 HMM 灵活）
      2. 可以使用任意特征（比最大熵模型更强大）
      3. 综合考虑整个序列的信息（比逐个标注更准确）
    """)

    # 演示 CRF vs HMM
    print("-" * 40)
    print("示例 1：CRF vs HMM 对比")
    explain_crf_vs_hmm()


def lesson_feature_template():
    """第二部分：特征模板设计"""

    print_separator("10.2 特征模板设计")

    print("""
    特征模板是 CRF 的核心。

    什么是特征模板？
      特征模板定义了"从数据中提取哪些特征"。

    生活类比：
      想象你在面试一个人，你的"面试问题清单"就是特征模板：
      1. 你的学历是什么？（当前词的特征）
      2. 你之前的工作是什么？（前一个词的特征）
      3. 你的期望薪资是多少？（当前位置的特征）

    常见的特征模板：
      1. Unigram 特征：当前词本身的属性
      2. Bigram 特征：当前词和前一个词的关系
      3. 词形特征：前缀、后缀、是否数字等
    """)

    # 演示英文特征提取
    print("-" * 40)
    print("示例 1：英文特征提取")

    en_sentence = ["John", "lives", "in", "New", "York"]
    print(f"\n  句子: {en_sentence}")

    features = extract_sentence_features(en_sentence, is_chinese=False)
    for i, (word, feat) in enumerate(zip(en_sentence, features)):
        print(f"\n  位置 {i}: '{word}'")
        for key, value in list(feat.items())[:6]:
            print(f"    {key}: {value}")

    # 演示中文特征提取
    print("\n" + "-" * 40)
    print("示例 2：中文特征提取")

    cn_sentence = ["小明", "在", "北京", "大学", "学习"]
    print(f"\n  句子: {cn_sentence}")

    features = extract_sentence_features(cn_sentence, is_chinese=True)
    for i, (word, feat) in enumerate(zip(cn_sentence, features)):
        print(f"\n  位置 {i}: '{word}'")
        for key, value in list(feat.items())[:6]:
            print(f"    {key}: {value}")


def lesson_simple_crf():
    """第三部分：简化 CRF 实现"""

    print_separator("10.3 简化 CRF 实现")

    print("""
    这里我们实现一个简化版的 CRF，帮助理解其核心原理。

    简化 CRF 包含三个核心组件：
      1. 特征提取：从数据中提取特征
      2. 权重计算：根据特征和权重计算得分
      3. 维特比解码：找到最优标签序列

    维特比算法：
      想象你在走一个迷宫：
      - 每一层有很多房间（可能的标签）
      - 每个房间有一个"得分"
      - 从一个房间走到下一个房间也有"得分"（转移得分）
      维特比算法就是找到一条"总得分最高"的路径。
    """)

    # 训练数据
    print("-" * 40)
    print("示例 1：训练简化 CRF")

    train_data = [
        (["小明", "去", "北京"], ["B-PER", "O", "B-LOC"]),
        (["小红", "在", "上海"], ["B-PER", "O", "B-LOC"]),
        (["今天", "天气", "好"], ["O", "O", "O"]),
        (["明天", "会", "下雨"], ["O", "O", "O"]),
        (["李华", "来自", "广东"], ["B-PER", "O", "B-LOC"]),
        (["王芳", "去", "广州"], ["B-PER", "O", "B-LOC"]),
    ]

    print(f"\n  训练数据: {len(train_data)} 个样本")
    for sent, labels in train_data:
        print(f"    {sent} → {labels}")

    # 提取特征
    X_train = [extract_sentence_features(s, is_chinese=True) for s, _ in train_data]
    y_train = [labels for _, labels in train_data]

    # 训练
    crf = SimpleCRF()
    crf.fit(X_train, y_train, epochs=30)

    # 预测
    print("\n" + "-" * 40)
    print("示例 2：使用 CRF 预测")

    test_cases = [
        ["小明", "去", "上海"],
        ["今天", "天气", "很好"],
        ["张三", "在", "北京"],
    ]

    for test_sent in test_cases:
        test_feat = extract_sentence_features(test_sent, is_chinese=True)
        pred = crf.viterbi(test_feat)
        print(f"\n  输入: {test_sent}")
        print(f"  预测: {pred}")


def lesson_sklearn_crfsuite():
    """第四部分：sklearn-crfsuite 实战"""

    print_separator("10.4 sklearn-crfsuite 实战")

    print("""
    在实际项目中，我们使用 sklearn-crfsuite。

    sklearn-crfsuite 的优点：
      1. 基于 CRFsuite，速度很快
      2. 支持多种优化算法（L-BFGS 等）
      3. 内置正则化，防止过拟合
      4. 与 sklearn 生态系统兼容

    安装方式：
      pip install sklearn-crfsuite
    """)

    # 演示 sklearn-crfsuite
    print("-" * 40)
    print("示例 1：sklearn-crfsuite 使用")
    demo_sklearn_crfsuite()

    # 演示特征重要性
    print("\n" + "-" * 40)
    print("示例 2：特征重要性分析")
    demo_crf_feature_importance()


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十章                        ║
    ║                                                      ║
    ║        ██████╗██████╗ ███████╗                       ║
    ║       ██╔════╝██╔══██╗██╔════╝                       ║
    ║       ██║     ██████╔╝█████╗                         ║
    ║       ██║     ██╔══██╗██╔══╝                         ║
    ║       ╚██████╗██║  ██║██║                            ║
    ║        ╚═════╝╚═╝  ╚═╝╚═╝                            ║
    ║                                                      ║
    ║              条 件 随 机 场                            ║
    ╚══════════════════════════════════════════════════════╝
    """)

    lesson_crf_concept()
    lesson_feature_template()
    lesson_simple_crf()
    lesson_sklearn_crfsuite()

    # 课程总结
    print("\n" + "=" * 60)
    print("  第十章 总结")
    print("=" * 60)
    print("""
    [OK] CRF 基本概念
         条件随机场是一种判别式模型
         综合考虑上下文信息，不依赖独立性假设
         比 HMM 更灵活，比最大熵模型更强大

    [OK] 特征模板设计
         Unigram 特征：当前词本身的属性
         Bigram 特征：当前词和前一个词的关系
         词形特征：前缀、后缀、是否数字等
         特征模板决定了 CRF 能学到什么

    [OK] 简化 CRF 实现
         感知机算法训练
         维特比算法解码
         理解 CRF 的核心原理

    [OK] sklearn-crfsuite 实战
         工业级 CRF 实现
         支持 L-BFGS 优化算法
         内置正则化，防止过拟合
    """)

    print("-" * 60)
    print("  下节预告：第十一章 — 新词发现与短语提取")
    print("-" * 60)
    print("""
    下一章我们将学习：
    - 互信息（Mutual Information）
    - 左右熵（Left/Right Entropy）
    - 新词发现算法
    - 短语提取算法

    这是无监督 NLP 的重要技术！
    """)
