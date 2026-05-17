import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第二章：中文分词 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 实现正向最大匹配分词
    2. 实现逆向最大匹配分词
    3. 构建简单的 HMM 分词器
    4. 使用 jieba 进行分词并比较不同模式

运行方式：
    python exercises.py
==============================================================================
"""


# ==============================================================================
# 练习 1：正向最大匹配分词
# ==============================================================================

def exercise_1_fmm(text: str, dictionary: set, max_len: int = 5) -> list:
    """
    练习 1：实现正向最大匹配分词

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你有一把尺子，从句子左边开始量。
    每次都量最长的那段，如果在词典里就切出来。
    如果不在，就量短一点，直到找到为止。

    ━━━━━━━ 提示 ━━━━━━━
    1. 创建空列表 result，当前位置 i = 0
    2. 当 i < len(text) 时循环：
       a. 从长度 max_len 到 1 尝试：
          - 截取 text[i:i+length]
          - 如果在 dictionary 中，加入 result，i += length，跳出内循环
       b. 如果都没匹配，把 text[i] 作为单字词，i += 1
    3. 返回 result

    参数：
        text: 待分词句子
        dictionary: 词典集合
        max_len: 最大词长

    返回：
        分词结果列表
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # result = []
    # i = 0
    # while i < len(text):
    #     matched = False
    #     for length in range(max_len, 0, -1):
    #         if i + length > len(text):
    #             continue
    #         word = text[i:i + length]
    #         if word in dictionary:
    #             result.append(word)
    #             i += length
    #             matched = True
    #             break
    #     if not matched:
    #         result.append(text[i])
    #         i += 1
    # return result
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：正向最大匹配分词")
    print("=" * 60)

    dictionary = {"今天", "天气", "真好", "我们", "去", "公园", "玩"}
    text = "今天天气真好"
    result = exercise_1_fmm(text, dictionary)

    if result is None:
        print("[未完成] 请实现 exercise_1_fmm 函数")
        return False

    expected = ["今天", "天气", "真好"]
    if result == expected:
        print(f"[正确] '{text}' → {result}")
        return True
    else:
        print(f"[错误] '{text}' → 期望 {expected}, 实际 {result}")
        return False


# ==============================================================================
# 练习 2：逆向最大匹配分词
# ==============================================================================

def exercise_2_bmm(text: str, dictionary: set, max_len: int = 5) -> list:
    """
    练习 2：实现逆向最大匹配分词

    ━━━━━━━ 生活类比 ━━━━━━━
    这次从蛋糕的右边开始切。也是每次切最大的一块。
    最后把结果反转（因为是从右往左收集的）。

    ━━━━━━━ 提示 ━━━━━━━
    1. 创建空列表 result，当前位置 i = len(text)
    2. 当 i > 0 时循环：
       a. 从长度 max_len 到 1 尝试：
          - start = i - length
          - 截取 text[start:i]
          - 如果在 dictionary 中，加入 result，i = start，跳出
       b. 如果都没匹配，把 text[i-1] 作为单字词，i -= 1
    3. result.reverse()，返回 result

    参数：
        text: 待分词句子
        dictionary: 词典集合
        max_len: 最大词长

    返回：
        分词结果列表（从左到右的顺序）
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # result = []
    # i = len(text)
    # while i > 0:
    #     matched = False
    #     for length in range(max_len, 0, -1):
    #         start = i - length
    #         if start < 0:
    #             continue
    #         word = text[start:i]
    #         if word in dictionary:
    #             result.append(word)
    #             i = start
    #             matched = True
    #             break
    #     if not matched:
    #         result.append(text[i - 1])
    #         i -= 1
    # result.reverse()
    # return result
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：逆向最大匹配分词")
    print("=" * 60)

    dictionary = {"今天", "天气", "真好", "我们", "去", "公园", "玩"}
    text = "今天天气真好"
    result = exercise_2_bmm(text, dictionary)

    if result is None:
        print("[未完成] 请实现 exercise_2_bmm 函数")
        return False

    expected = ["今天", "天气", "真好"]
    if result == expected:
        print(f"[正确] '{text}' → {result}")
        return True
    else:
        print(f"[错误] '{text}' → 期望 {expected}, 实际 {result}")
        return False


# ==============================================================================
# 练习 3：使用 jieba 分词
# ==============================================================================

def exercise_3_jieba_compare(text: str) -> dict:
    """
    练习 3：使用 jieba 的三种模式分词，并比较结果

    ━━━━━━━ 提示 ━━━━━━━
    1. 导入 jieba
    2. 用 jieba.cut(text, cut_all=False) 得到精确模式结果
    3. 用 jieba.cut(text, cut_all=True) 得到全模式结果
    4. 用 jieba.cut_for_search(text) 得到搜索引擎模式结果
    5. 返回字典 {"精确模式": list1, "全模式": list2, "搜索引擎模式": list3}

    参数：
        text: 待分词句子

    返回：
        包含三种模式结果的字典
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # import jieba
    # return {
    #     "精确模式": list(jieba.cut(text, cut_all=False)),
    #     "全模式": list(jieba.cut(text, cut_all=True)),
    #     "搜索引擎模式": list(jieba.cut_for_search(text)),
    # }
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：jieba 三种模式分词比较")
    print("=" * 60)

    try:
        import jieba
    except ImportError:
        print("[跳过] jieba 未安装，请运行: pip install jieba")
        return False

    text = "我来到北京清华大学"
    result = exercise_3_jieba_compare(text)

    if result is None:
        print("[未完成] 请实现 exercise_3_jieba_compare 函数")
        return False

    print(f"测试文本: '{text}'")
    for mode, words in result.items():
        print(f"  {mode}: {' / '.join(words)}")

    # 检查结果是否合理
    if "精确模式" in result and "全模式" in result:
        print("[正确] 三种模式结果已生成")
        return True
    else:
        print("[错误] 结果格式不正确")
        return False


# ==============================================================================
# 练习 4：分词评估
# ==============================================================================

def exercise_4_evaluate(predicted: list, golden: list) -> dict:
    """
    练习 4：计算分词的准确率、召回率和 F1 值

    ━━━━━━━ 提示 ━━━━━━━
    1. 将 predicted 和 golden 转为集合
    2. 计算交集（correct = pred_set & gold_set）
    3. precision = len(correct) / len(pred_set)
    4. recall = len(correct) / len(gold_set)
    5. f1 = 2 * precision * recall / (precision + recall)
    6. 返回 {"precision": ..., "recall": ..., "f1": ...}

    参数：
        predicted: 模型分词结果
        golden: 标准答案

    返回：
        包含三个指标的字典
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # pred_set = set(predicted)
    # gold_set = set(golden)
    # correct = pred_set & gold_set
    # precision = len(correct) / len(pred_set) if pred_set else 0
    # recall = len(correct) / len(gold_set) if gold_set else 0
    # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    pass


def test_exercise_4():
    """测试练习 4"""
    print("\n" + "=" * 60)
    print("练习 4：分词评估")
    print("=" * 60)

    golden = ["今天", "天气", "真好"]
    predicted = ["今天", "天气", "真好"]
    result = exercise_4_evaluate(predicted, golden)

    if result is None:
        print("[未完成] 请实现 exercise_4_evaluate 函数")
        return False

    if result.get("f1") == 1.0:
        print(f"[正确] 完美匹配: P={result['precision']}, R={result['recall']}, F1={result['f1']}")

        # 测试不完美匹配
        predicted2 = ["今天", "天气真", "好"]
        result2 = exercise_4_evaluate(predicted2, golden)
        print(f"       不完美匹配: P={result2['precision']}, R={result2['recall']}, F1={result2['f1']}")
        return True
    else:
        print(f"[错误] 完美匹配应该 F1=1.0, 实际 F1={result.get('f1')}")
        return False


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第二章 练习                    ║
    ║        中文分词                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    results = []
    results.append(("练习1: 正向最大匹配", test_exercise_1()))
    results.append(("练习2: 逆向最大匹配", test_exercise_2()))
    results.append(("练习3: jieba三种模式", test_exercise_3()))
    results.append(("练习4: 分词评估", test_exercise_4()))

    print("\n" + "=" * 60)
    print("  练习清单")
    print("=" * 60)
    for name, passed in results:
        status = "[完成]" if passed else "[未完成]"
        print(f"  {status} {name}")

    completed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  完成率: {completed}/{total}")

    if completed == total:
        print("\n  所有练习完成！你已经掌握了中文分词的核心技术。")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
