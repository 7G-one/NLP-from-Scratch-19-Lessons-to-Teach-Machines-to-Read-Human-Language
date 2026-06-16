import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
==============================================================================
第六章：语义角色标注 — 练习题
==============================================================================
G-one NLP 学院
日期：2026-05-16

本章练习：
    1. 识别语义角色
    2. 构建 SRL 结果
    3. 提取事件三元组

运行方式：
    python exercises.py

提示：
    - 每个练习都有详细的提示，按照提示一步步来
    - 先自己写，写不出来再看注释中的参考答案
    - 运行后会自动检查你的答案是否正确
==============================================================================
"""

from srl import SRLArgument, SRLResult


# ==============================================================================
# 练习 1：识别语义角色
# ==============================================================================

def exercise_1_identify_role(word: str, pos: str, position: str,
                             has_loc_keyword: bool = False) -> str:
    """
    练习 1：根据词的特征判断它的语义角色

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你是一个记者，看到一个事件中的各种人和物，
    你需要判断每个人扮演什么角色：
    - 谁是"做事情的人"？→ 施事
    - 谁是"被做的事情"？→ 受事
    - "什么时候"发生的？→ 时间
    - "在哪里"发生的？→ 地点

    ━━━━━━━ 提示 ━━━━━━━
    根据以下规则判断：
    - 如果 pos 是 "t"（时间词）→ "TMP"
    - 如果 pos 是 "ns" 或 "s"（地点词）→ "LOC"
    - 如果 pos 是 "r"（代词）且 position 是 "left" → "A0"（施事）
    - 如果 pos 是 "nr"（人名）且 position 是 "left" → "A0"（施事）
    - 如果 pos 是 "n"（名词）且 position 是 "right" → "A1"（受事）
    - 如果 has_loc_keyword 为 True → "LOC"
    - 其他情况 → "DEP"（未分类）

    参数：
        word: 词
        pos: 词性
        position: 在谓词的 "left" 还是 "right"
        has_loc_keyword: 前面是否有"在"、"从"等介词

    返回：
        语义角色标签
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # if pos == "t":
    #     return "TMP"
    # if pos in ("ns", "s"):
    #     return "LOC"
    # if has_loc_keyword:
    #     return "LOC"
    # if pos == "r" and position == "left":
    #     return "A0"
    # if pos == "nr" and position == "left":
    #     return "A0"
    # if pos == "n" and position == "right":
    #     return "A1"
    # return "DEP"
    pass


def test_exercise_1():
    """测试练习 1"""
    print("\n" + "=" * 60)
    print("练习 1：识别语义角色")
    print("=" * 60)

    test_cases = [
        # (word, pos, position, has_loc_keyword, expected)
        ("昨天", "t", "left", False, "TMP"),
        ("北京", "ns", "left", False, "LOC"),
        ("我", "r", "left", False, "A0"),
        ("小明", "nr", "left", False, "A0"),
        ("苹果", "n", "right", False, "A1"),
        ("学校", "n", "left", True, "LOC"),
        ("快速", "d", "left", False, "DEP"),
    ]

    all_correct = True
    for word, pos, position, has_loc, expected in test_cases:
        result = exercise_1_identify_role(word, pos, position, has_loc)
        if result is None:
            print("[未完成] 请实现 exercise_1_identify_role 函数")
            return False
        if result == expected:
            print(f"  [正确] \"{word}\" ({pos}, {position}) → {result}")
        else:
            print(f"  [错误] \"{word}\" ({pos}, {position}) → 期望 {expected}, 实际 {result}")
            all_correct = False

    return all_correct


# ==============================================================================
# 练习 2：构建 SRL 结果
# ==============================================================================

def exercise_2_build_srl(predicate: str, args_data: list) -> SRLResult:
    """
    练习 2：根据给定的谓词和论元数据，构建 SRLResult 对象

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你在填写一份事件报告：
    - 谓词：发生了什么事件
    - 论元：事件的各个参与者（用标签和内容描述）

    ━━━━━━━ 提示 ━━━━━━━
    1. 创建一个 SRLResult 对象，predicate 设为给定的谓词
    2. 遍历 args_data，每个元素是 (标签, 文本)
    3. 用 SRLArgument(tag, text) 创建论元对象
    4. 用 result.add_argument(arg) 添加到结果中
    5. 返回 SRLResult 对象

    参数：
        predicate: 谓词
        args_data: [(标签, 文本), ...] 列表

    返回：
        SRLResult 对象
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # result = SRLResult(predicate=predicate)
    # for tag, text in args_data:
    #     arg = SRLArgument(tag=tag, text=text)
    #     result.add_argument(arg)
    # return result
    pass


def test_exercise_2():
    """测试练习 2"""
    print("\n" + "=" * 60)
    print("练习 2：构建 SRL 结果")
    print("=" * 60)

    predicate = "踢"
    args_data = [("A0", "小明"), ("A1", "小红"), ("TMP", "昨天")]

    result = exercise_2_build_srl(predicate, args_data)

    if result is None:
        print("[未完成] 请实现 exercise_2_build_srl 函数")
        return False

    # 检查谓词
    if result.predicate != "踢":
        print(f"[错误] 谓词应为 \"踢\"，实际为 \"{result.predicate}\"")
        return False

    # 检查论元数量
    if len(result.arguments) != 3:
        print(f"[错误] 应有 3 个论元，实际有 {len(result.arguments)} 个")
        return False

    # 检查每个论元
    agent = result.get_argument("A0")
    patient = result.get_argument("A1")
    time = result.get_argument("TMP")

    if not agent or agent.text != "小明":
        print(f"[错误] A0 应为 \"小明\"")
        return False
    if not patient or patient.text != "小红":
        print(f"[错误] A1 应为 \"小红\"")
        return False
    if not time or time.text != "昨天":
        print(f"[错误] TMP 应为 \"昨天\"")
        return False

    print(f"[正确] SRL 结果：")
    result.display()
    return True


# ==============================================================================
# 练习 3：提取事件三元组
# ==============================================================================

def exercise_3_extract_triple(srl_result: SRLResult) -> tuple:
    """
    练习 3：从 SRL 结果中提取事件三元组

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你在读新闻标题，需要提取最核心的三个信息：
    - 谁（施事）
    - 做了什么（动作）
    - 对谁（受事）

    ━━━━━━━ 提示 ━━━━━━━
    1. 从 srl_result 中获取 A0（施事）论元
    2. 从 srl_result 中获取 A1（受事）论元
    3. 获取谓词
    4. 如果 A0 存在，取其 text，否则用 "未知"
    5. 如果 A1 存在，取其 text，否则用 "未知"
    6. 返回 (施事文本, 谓词, 受事文本) 三元组

    参数：
        srl_result: SRLResult 对象

    返回：
        (施事, 动作, 受事) 三元组
    """
    # TODO: 请在这里写你的代码
    #
    # 参考答案：
    # agent = srl_result.get_argument("A0")
    # patient = srl_result.get_argument("A1")
    # agent_text = agent.text if agent else "未知"
    # patient_text = patient.text if patient else "未知"
    # return (agent_text, srl_result.predicate, patient_text)
    pass


def test_exercise_3():
    """测试练习 3"""
    print("\n" + "=" * 60)
    print("练习 3：提取事件三元组")
    print("=" * 60)

    # 测试用例 1：完整的三元组
    result1 = SRLResult(predicate="吃")
    result1.add_argument(SRLArgument("A0", "小明"))
    result1.add_argument(SRLArgument("A1", "苹果"))

    triple1 = exercise_3_extract_triple(result1)
    if triple1 is None:
        print("[未完成] 请实现 exercise_3_extract_triple 函数")
        return False

    if triple1 == ("小明", "吃", "苹果"):
        print(f"  [正确] {triple1}")
    else:
        print(f"  [错误] 期望 ('小明', '吃', '苹果'), 实际 {triple1}")
        return False

    # 测试用例 2：缺少受事
    result2 = SRLResult(predicate="跑")
    result2.add_argument(SRLArgument("A0", "他"))

    triple2 = exercise_3_extract_triple(result2)
    if triple2 == ("他", "跑", "未知"):
        print(f"  [正确] {triple2}")
    else:
        print(f"  [错误] 期望 ('他', '跑', '未知'), 实际 {triple2}")
        return False

    # 测试用例 3：缺少施事
    result3 = SRLResult(predicate="发布")
    result3.add_argument(SRLArgument("A1", "新手机"))

    triple3 = exercise_3_extract_triple(result3)
    if triple3 == ("未知", "发布", "新手机"):
        print(f"  [正确] {triple3}")
    else:
        print(f"  [错误] 期望 ('未知', '发布', '新手机'), 实际 {triple3}")
        return False

    return True


# ==============================================================================
# 主程序：运行所有练习测试
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第六章 练习                    ║
    ║        语义角色标注                                   ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 运行所有练习测试
    results = []
    results.append(("练习1: 识别语义角色", test_exercise_1()))
    results.append(("练习2: 构建SRL结果", test_exercise_2()))
    results.append(("练习3: 提取事件三元组", test_exercise_3()))

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
        print("  你已经掌握了语义角色标注的基础。")
        print("  基础篇（第 1-6 章）的学习到此结束！")
        print("  接下来将进入核心技术篇。")
    else:
        print(f"\n  还有 {total - completed} 个练习未完成。")
        print("  不要着急，语义角色标注需要多练习才能理解。")
        print("  如果实在写不出来，可以查看注释中的参考答案。")
