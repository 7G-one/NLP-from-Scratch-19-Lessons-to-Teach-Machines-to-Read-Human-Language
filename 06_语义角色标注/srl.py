"""
==============================================================================
第六章：语义角色标注（Semantic Role Labeling, SRL）
==============================================================================
日期：2026-05-16

同学们好！前面我们学了依存句法分析，知道了"谁修饰谁"。
今天我们再深入一层，学习语义角色标注 —— 弄清楚"谁对谁做了什么"。

----------------------------------------------------------------------
生活类比：语义角色标注就像记者写新闻的 5W1H
----------------------------------------------------------------------

想象你是一个记者，看到一条新闻："小明昨天在学校用脚踢了小红"

你需要回答这些问题：
  - WHO（谁）:     小明（施事/Agent）
  - DID WHAT（做了什么）: 踢（谓词/Predicate）
  - TO WHOM（对谁）: 小红（受事/Patient）
  - WHEN（什么时候）: 昨天（时间/Time）
  - WHERE（在哪里）: 在学校（地点/Location）
  - HOW（怎么踢的）: 用脚（方式/Manner）

这就是语义角色标注（SRL）要做的事！

  ┌──────────────────────────────────────────────┐
  │  小明   昨天   在学校   用脚   踢   小红      │
  │  │      │      │       │     │    │         │
  │  A0     TMP    LOC     MNR   V    A1        │
  │  施事   时间   地点     方式   谓词  受事      │
  └──────────────────────────────────────────────┘

----------------------------------------------------------------------
什么是语义角色？
----------------------------------------------------------------------

语义角色描述的是一个事件中各个参与者扮演的角色。
不同于语法角色（主语、宾语），语义角色关注的是"意义"。

  ┌──────────────────────────────────────────────┐
  │  语法 vs 语义                                 │
  ├──────────────────┬───────────────────────────┤
  │  语法角色         │  语义角色                   │
  │  (Syntactic)      │  (Semantic)                │
  ├──────────────────┼───────────────────────────┤
  │  主语 (Subject)   │  施事 (Agent) — 谁做的     │
  │  宾语 (Object)    │  受事 (Patient) — 对谁做的  │
  │  修饰语           │  时间、地点、方式等          │
  └──────────────────┴───────────────────────────┘

  例："门被风吹开了"
    语法主语：门
    语义受事：门（被吹开的）
    语义施事：风（吹的执行者）

  语法角色和语义角色不一定对应！这正是 SRL 的价值。

----------------------------------------------------------------------
常见的语义角色（PropBank 标注体系）
----------------------------------------------------------------------

  ┌──────────────────────────────────────────────────┐
  │  标签    │  含义            │  说明                │
  ├─────────┼─────────────────┼────────────────────┤
  │  V       │  谓词（动词）     │  事件的核心           │
  │  A0      │  施事（Agent）    │  动作的执行者         │
  │  A1      │  受事（Patient）  │  动作的承受者         │
  │  A2      │  间接受事/工具    │  给予的对象等         │
  │  TMP     │  时间（Time）    │  什么时候发生         │
  │  LOC     │  地点（Location）│  在哪里发生           │
  │  MNR     │  方式（Manner）  │  怎样发生的           │
  │  CAU     │  原因（Cause）   │  为什么发生           │
  │  PRP     │  目的（Purpose） │  为了什么             │
  │  DIR     │  方向（Direction）│  向哪里              │
  │  EXT     │  程度（Extent）  │  到什么程度           │
  │  DIS     │  话语标记        │  连接词等             │
  └─────────┴─────────────────┴────────────────────┘

==============================================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')


# ==============================================================================
# 第一部分：语义角色的数据结构
# ==============================================================================
#
# 我们用简单的类来表示语义角色标注的结果。
#
# 一次 SRL 的结果包含：
#   - 谓词（Predicate）：事件的核心动词
#   - 多个论元（Arguments）：事件的参与者
#
# 就像新闻报道中的要素：
#   - 事件：发生了什么（谓词）
#   - 各种参与者：谁、对谁、在哪、什么时候（论元）
#
# ==============================================================================

class SRLArgument:
    """
    语义角色论元

    ━━━━━━━ 生活类比 ━━━━━━━
    想象新闻中的一个要素：
    - tag: 要素类型（如 WHO、WHERE）
    - text: 具体内容（如 "小明"、"在学校"）
    - start: 内容在句子中的起始位置
    - end: 内容在句子中的结束位置

    属性：
        tag: 语义角色标签（如 A0、A1、TMP、LOC）
        text: 论元的文本内容
        start: 在词列表中的起始索引
        end: 在词列表中的结束索引（不包含）
    """

    def __init__(self, tag: str, text: str, start: int = 0, end: int = 0):
        """
        初始化一个语义角色论元

        参数：
            tag: 语义角色标签
            text: 论元文本
            start: 起始索引
            end: 结束索引
        """
        self.tag = tag
        self.text = text
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Arg({self.tag}: \"{self.text}\")"


class SRLResult:
    """
    一次语义角色标注的结果

    ━━━━━━━ 生活类比 ━━━━━━━
    想象一篇新闻报道的要素清单：
    - 谓词：事件是什么（如"踢"）
    - 论元列表：事件的各个参与者和背景信息

    属性：
        predicate: 谓词（动词）
        predicate_index: 谓词在词列表中的索引
        arguments: 语义角色论元列表
        words: 原始词列表
    """

    def __init__(self, predicate: str, predicate_index: int = 0,
                 arguments: list = None, words: list = None):
        """
        初始化 SRL 结果

        参数：
            predicate: 谓词
            predicate_index: 谓词在词列表中的索引
            arguments: SRLArgument 对象的列表
            words: 原始词列表
        """
        self.predicate = predicate
        self.predicate_index = predicate_index
        self.arguments = arguments or []
        self.words = words or []

    def add_argument(self, arg: SRLArgument):
        """添加一个论元"""
        self.arguments.append(arg)

    def get_argument(self, tag: str) -> SRLArgument:
        """
        根据标签获取论元

        参数：
            tag: 语义角色标签

        返回：
            SRLArgument 对象，如果没有找到返回 None
        """
        for arg in self.arguments:
            if arg.tag == tag:
                return arg
        return None

    def display(self):
        """可视化显示 SRL 结果"""
        print(f"  谓词: {self.predicate}")
        print(f"  {'─' * 40}")
        for arg in self.arguments:
            print(f"  {arg.tag:<8} → {arg.text}")

    def to_triple(self) -> tuple:
        """
        转换为三元组（施事, 谓词, 受事）

        返回：
            (施事, 谓词, 受事) 三元组
        """
        agent = self.get_argument("A0")
        patient = self.get_argument("A1")
        agent_text = agent.text if agent else "未知"
        patient_text = patient.text if patient else "未知"
        return (agent_text, self.predicate, patient_text)


# ==============================================================================
# 第二部分：谓词-论元结构
# ==============================================================================
#
# 谓词-论元结构（Predicate-Argument Structure）是 SRL 的核心概念。
#
# 生活类比：就像一个舞台剧：
#   - 谓词 = 剧情（发生了什么事）
#   - 论元 = 演员（谁参与了这件事）
#   - 每个演员都有一个角色（施事、受事等）
#
# ==============================================================================

# 常见的语义角色标签定义
SRL_ROLES = {
    "V":   "谓词（动词）",
    "A0":  "施事 — 动作的执行者",
    "A1":  "受事 — 动作的承受者",
    "A2":  "间接受事 — 给予的对象、工具等",
    "TMP": "时间 — 什么时候",
    "LOC": "地点 — 在哪里",
    "MNR": "方式 — 怎样做",
    "CAU": "原因 — 为什么",
    "PRP": "目的 — 为了什么",
    "DIR": "方向 — 向哪里",
    "EXT": "程度 — 到什么程度",
    "DIS": "话语标记 — 连接词",
}


def explain_roles():
    """打印语义角色的详细说明"""
    print("  常见语义角色标签：")
    print(f"  {'标签':<8}{'含义':<10}{'说明'}")
    print(f"  {'─' * 50}")
    for tag, desc in SRL_ROLES.items():
        print(f"  {tag:<8}{desc}")


# ==============================================================================
# 第三部分：基于规则的语义角色标注器
# ==============================================================================
#
# 最简单的 SRL 方法：用规则来判断论元的角色。
#
# 核心思想：
#   1. 找到句子中的谓词（动词）
#   2. 谓词左边的名词/代词 → 施事（A0）
#   3. 谓词右边的名词 → 受事（A1）
#   4. 时间词 → 时间角色（TMP）
#   5. 处所词 → 地点角色（LOC）
#   6. 副词 → 方式角色（MNR）
#
# 就像读新闻：
#   - "谁"在动词左边 → 施事
#   - "什么"在动词右边 → 受事
#   - "昨天、今天" → 时间
#   - "在学校、在公园" → 地点
#
# ==============================================================================

class RuleBasedSRL:
    """
    基于规则的语义角色标注器

    ━━━━━━━ 核心思想 ━━━━━━━
    1. 找到谓词（动词）
    2. 谓词左边的代词/人名 → 施事 (A0)
    3. 谓词右边的名词 → 受事 (A1)
    4. 时间词 → 时间 (TMP)
    5. 处所词/介词+名词 → 地点 (LOC)
    6. 副词 → 方式 (MNR)
    """

    def __init__(self):
        """初始化规则标注器"""
        # 谓词通常是动词
        self.verb_pos = {"v"}

        # 施事通常由代词或人名充当
        self.agent_pos = {"r", "nr"}

        # 受事通常由名词充当
        self.patient_pos = {"n", "nr", "ns", "nt"}

        # 时间角色
        self.time_pos = {"t"}
        self.time_words = {"昨天", "今天", "明天", "上午", "下午", "晚上",
                           "去年", "今年", "明年", "刚才", "现在", "以前"}

        # 地点角色
        self.loc_pos = {"s", "ns"}
        self.loc_keywords = {"在", "从", "到"}

        # 方式角色
        self.manner_pos = {"d"}  # 副词

    def parse(self, words: list, pos_tags: list) -> list:
        """
        对句子进行语义角色标注

        ━━━━━━━ 算法步骤 ━━━━━━━
        1. 找到句子中的所有谓词（动词）
        2. 对每个谓词，分析其周围的论元
        3. 根据词性和位置判断论元的语义角色

        参数：
            words: 词列表
            pos_tags: 词性列表

        返回：
            SRLResult 对象的列表（每个谓词一个结果）
        """
        results = []

        # 第一步：找到所有谓词（动词）
        verb_indices = []
        for i, pos in enumerate(pos_tags):
            if pos in self.verb_pos:
                verb_indices.append(i)

        # 如果没有动词，返回空
        if not verb_indices:
            return results

        # 第二步：对每个谓词分析论元
        for verb_idx in verb_indices:
            result = self._analyze_predicate(words, pos_tags, verb_idx)
            results.append(result)

        return results

    def _analyze_predicate(self, words: list, pos_tags: list,
                           verb_idx: int) -> SRLResult:
        """
        分析单个谓词的论元结构

        参数：
            words: 词列表
            pos_tags: 词性列表
            verb_idx: 谓词的索引

        返回：
            SRLResult 对象
        """
        verb = words[verb_idx]
        result = SRLResult(predicate=verb, predicate_index=verb_idx, words=words)

        # 遍历句子中的每个词，判断它是什么角色
        i = 0
        while i < len(words):
            if i == verb_idx:
                # 跳过谓词本身
                i += 1
                continue

            word = words[i]
            pos = pos_tags[i]

            # 判断角色
            role = self._classify_role(word, pos, i, verb_idx, words, pos_tags)

            if role:
                # 尝试合并连续的同角色词（如 "北京大学"）
                text = word
                end = i + 1
                # 简单合并：如果下一个词也是名词且没有被其他规则匹配
                while (end < len(words) and
                       end != verb_idx and
                       pos_tags[end] in ("n", "ns", "nt") and
                       self._classify_role(words[end], pos_tags[end],
                                           end, verb_idx, words, pos_tags) == role):
                    text += words[end]
                    end += 1

                arg = SRLArgument(tag=role, text=text, start=i, end=end)
                result.add_argument(arg)
                i = end
            else:
                i += 1

        return result

    def _classify_role(self, word: str, pos: str, idx: int,
                       verb_idx: int, words: list, pos_tags: list) -> str:
        """
        分类一个词的语义角色

        ━━━━━━━ 生活类比 ━━━━━━━
        就像判断一个人在一部电影中扮演什么角色：
        - 看他的位置（在主角前面还是后面）
        - 看他的身份（是人名还是地名）
        - 看他说的话（有没有"在"、"用"等关键词）
        """
        # 时间词 → TMP
        if pos in self.time_pos or word in self.time_words:
            return "TMP"

        # 地点词 → LOC
        if pos in self.loc_pos:
            return "LOC"

        # 介词 + 地点词 → LOC（如 "在学校"）
        if pos == "p" and word in self.loc_keywords:
            # 检查后面是否跟着地点名词
            if idx + 1 < len(words) and pos_tags[idx + 1] in ("n", "ns", "s"):
                return "LOC"

        # 谓词左边的代词/人名 → A0（施事）
        if idx < verb_idx and pos in self.agent_pos:
            return "A0"

        # 谓词左边的普通名词也可能是施事
        if idx < verb_idx and pos in ("n",) and self._likely_agent(word):
            return "A0"

        # 谓词右边的名词 → A1（受事）
        if idx > verb_idx and pos in self.patient_pos:
            return "A1"

        # 副词 → MNR（方式）
        if pos in self.manner_pos:
            return "MNR"

        # 介词短语的特殊处理
        if pos == "p" and word == "用":
            # "用" 后面的词通常是工具/方式
            return "MNR"

        return None

    def _likely_agent(self, word: str) -> bool:
        """
        判断一个词是否可能是施事

        ━━━━━━━ 生活类比 ━━━━━━━
        就像判断一个人是否可能是"做事情的人"：
        - 人称代词（我、你、他）通常是施事
        - 普通名词需要更多上下文判断
        """
        agent_words = {"我", "你", "他", "她", "它", "我们", "你们", "他们",
                       "大家", "人们", "谁"}
        return word in agent_words


# ==============================================================================
# 第四部分：SRL 结果的三元组提取
# ==============================================================================
#
# SRL 的一个重要应用是提取"事件三元组"：
#   （施事, 动作, 受事）
#
# 就像新闻标题：
#   "小明踢了小红" → (小明, 踢, 小红)
#   "苹果公司发布了新手机" → (苹果公司, 发布, 新手机)
#
# ==============================================================================

def extract_triples(words: list, pos_tags: list) -> list:
    """
    从句子中提取事件三元组

    ━━━━━━━ 生活类比 ━━━━━━━
    就像从新闻中提取关键信息：
    - 谁（施事）做了什么（动作）对谁（受事）

    参数：
        words: 词列表
        pos_tags: 词性列表

    返回：
        [(施事, 动作, 受事), ...] 三元组列表
    """
    parser = RuleBasedSRL()
    srl_results = parser.parse(words, pos_tags)

    triples = []
    for result in srl_results:
        triple = result.to_triple()
        # 只保留有施事和受事的三元组
        if triple[0] != "未知" and triple[2] != "未知":
            triples.append(triple)

    return triples


# ==============================================================================
# 第五部分：演示函数
# ==============================================================================

def demo_srl_concept():
    """演示语义角色标注的基本概念"""

    print("=" * 60)
    print("语义角色标注基本概念")
    print("=" * 60)

    print("""
    语义角色标注（SRL）回答的核心问题：
    ┌──────────────────────────────────────────────┐
    │  "谁" 对 "谁" 做了 "什么"                      │
    │  "什么时候" "在哪里" "怎么做的"                  │
    └──────────────────────────────────────────────┘

    例句："小明昨天在学校踢了小红"

    ┌──────────────────────────────────────────────┐
    │  小明  昨天  在学校  踢  小红                    │
    │  │     │     │     │  │                       │
    │  A0   TMP   LOC    V  A1                       │
    │  施事  时间  地点  谓词 受事                     │
    └──────────────────────────────────────────────┘

    三元组提取：(小明, 踢, 小红)
    """)

    explain_roles()


def demo_rule_based_srl():
    """演示基于规则的 SRL"""

    print("=" * 60)
    print("基于规则的语义角色标注")
    print("=" * 60)

    parser = RuleBasedSRL()

    test_cases = [
        (["小明", "在", "公园", "踢", "足球"], ["nr", "p", "n", "v", "n"]),
        (["我", "昨天", "买", "了", "一本", "书"], ["r", "t", "v", "u", "m", "n"]),
        (["苹果公司", "在", "北京", "发布", "新手机"], ["nt", "p", "ns", "v", "n"]),
        (["他", "用", "筷子", "吃", "面条"], ["r", "p", "n", "v", "n"]),
    ]

    for words, pos_tags in test_cases:
        sentence = "".join(words)
        print(f"\n句子: {sentence}")

        results = parser.parse(words, pos_tags)
        if results:
            for r in results:
                r.display()
                triple = r.to_triple()
                print(f"  三元组: {triple}")
        else:
            print("  (未识别到谓词)")


def demo_triple_extraction():
    """演示事件三元组提取"""

    print("=" * 60)
    print("事件三元组提取")
    print("=" * 60)

    print("""
    三元组 = (施事, 动作, 受事)

    这是信息抽取的基础：
    - 从新闻中提取事件
    - 从对话中提取意图
    - 从文档中构建知识图谱
    """)

    test_cases = [
        (["小明", "吃", "苹果"], ["nr", "v", "n"]),
        (["我", "喜欢", "学习"], ["r", "v", "v"]),
        (["老师", "在", "教室", "讲课"], ["n", "p", "n", "v"]),
    ]

    for words, pos_tags in test_cases:
        sentence = "".join(words)
        triples = extract_triples(words, pos_tags)
        print(f"\n  句子: {sentence}")
        if triples:
            for t in triples:
                print(f"  三元组: {t}")
        else:
            print(f"  三元组: (未提取到完整三元组)")


def demo_srl_vs_dependency():
    """对比依存分析和语义角色标注"""

    print("=" * 60)
    print("依存分析 vs 语义角色标注")
    print("=" * 60)

    print("""
    ┌──────────────────────────────────────────────────┐
    │            依存分析           语义角色标注          │
    ├──────────────────────────────────────────────────┤
    │  分析层次    句法层面          语义层面             │
    │  关注点      谁修饰谁          谁对谁做了什么        │
    │  输出        依存树            谓词-论元结构         │
    │  关系类型    SBV/VOB/ATT...   A0/A1/TMP/LOC...   │
    │  应用        句子结构分析      信息抽取/问答         │
    └──────────────────────────────────────────────────┘

    例句："小明在学校踢足球"

    依存分析：
      小明 --SBV--> 踢
      在 --POB--> 学校
      踢 --HED--> ROOT
      足球 --VOB--> 踢

    语义角色标注：
      谓词: 踢
      A0 (施事): 小明
      LOC (地点): 在学校
      A1 (受事): 足球

    关键区别：
    - 依存分析看的是"句法结构"
    - SRL 看的是"语义含义"
    - "门被风吹开了"：依存分析看语法，SRL 能识别"风"是施事
    """)


def demo_application_scenarios():
    """演示 SRL 的应用场景"""

    print("=" * 60)
    print("语义角色标注的应用场景")
    print("=" * 60)

    print("""
    SRL 在 NLP 中有很多重要应用：

    1. 信息抽取
       ┌──────────────────────────────────────────┐
       │  新闻："华为在2024年发布了新手机Mate70"      │
       │  SRL 提取：                                │
       │    施事：华为                               │
       │    动作：发布                               │
       │    受事：新手机Mate70                       │
       │    时间：2024年                             │
       └──────────────────────────────────────────┘

    2. 问答系统
       ┌──────────────────────────────────────────┐
       │  问题："谁踢了小红？"                       │
       │  SRL 分析句子后，找到 A0（施事）→ 回答       │
       └──────────────────────────────────────────┘

    3. 知识图谱构建
       ┌──────────────────────────────────────────┐
       │  文本 → SRL → 三元组 → 知识图谱             │
       │  (小明, 就读于, 北京大学)                    │
       └──────────────────────────────────────────┘

    4. 情感分析
       ┌──────────────────────────────────────────┐
       │  "这家餐厅的菜很好吃，但服务太差了"           │
       │  SRL 识别情感对象：                         │
       │    A1: 菜 → 正面情感                       │
       │    A1: 服务 → 负面情感                      │
       └──────────────────────────────────────────┘
    """)


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第六章                        ║
    ║        语义角色标注                                   ║
    ╚══════════════════════════════════════════════════════╝
    """)

    demo_srl_concept()
    demo_rule_based_srl()
    demo_triple_extraction()
    demo_srl_vs_dependency()
    demo_application_scenarios()

    print("\n" + "=" * 60)
    print("第六章 总结")
    print("=" * 60)
    print("""
    [OK] 什么是语义角色标注 — 分析"谁对谁做了什么"
    [OK] 语义角色类型 — A0(施事)、A1(受事)、TMP(时间)等
    [OK] 谓词-论元结构 — 谓词 + 多个论元
    [OK] 基于规则的 SRL — 用词性规则判断语义角色
    [OK] 三元组提取 — (施事, 动作, 受事)
    [OK] 应用场景 — 信息抽取、问答、知识图谱
    """)

    # =============================================
    # 下节课预告
    # =============================================
    """
    下节课我们将学习文本相似度（Text Similarity）：
    - 余弦相似度 —— 用向量夹角衡量两段文本的相似程度
    - Jaccard 相似度 —— 用集合重叠比例衡量相似性
    - 编辑距离 —— 衡量两个字符串之间的"变换代价"
    """


# ==============================================================================
# 第六部分：基于 BIO 序列标注的语义角色标注
# ==============================================================================
#
# 前面我们用规则来做 SRL，现在我们学习一种更强大的方法：
# 序列标注（Sequence Labeling）。
#
# 核心思想：
#   把 SRL 问题转化为"给每个词打标签"的问题。
#   就像给文章中的每个词贴一个"角色标签"。
#
# 生活类比：
#   想象你在看一部电影，给每个演员贴标签：
#   - "小明" → 主角（B-A0）
#   - "昨天" → 时间背景（B-TMP）
#   - "在学校" → 场景（B-LOC）
#   - "踢" → 核心动作（V）
#   - "小红" → 配角（B-A1）
#
# BIO 标注体系：
#   B-xxx = Begin，表示 xxx 角色的开始
#   I-xxx = Inside，表示 xxx 角色的延续
#   O     = Outside，表示不属于任何角色
#
#   例："小明在学校踢足球"
#     小明 → B-A0    (施事的开始)
#     在  → B-LOC   (地点的开始)
#     学校 → I-LOC   (地点的延续)
#     踢  → V       (谓词)
#     足球 → B-A1    (受事的开始)
#
# ==============================================================================

import random as _random
from collections import defaultdict as _defaultdict


# 定义所有可能的 BIO 标签
# ━━━━━━━ 生活类比 ━━━━━━━
# 就像一套完整的"角色标签系统"：
# - A0 = 主角（施事）
# - A1 = 配角（受事）
# - TMP = 时间背景
# - LOC = 场景地点
BIO_TAGS = [
    "B-A0", "I-A0",     # 施事（Agent）— 动作的执行者
    "B-A1", "I-A1",     # 受事（Patient）— 动作的承受者
    "B-A2", "I-A2",     # 间接受事 — 给予对象、工具等
    "B-TMP", "I-TMP",   # 时间（Time）— 什么时候
    "B-LOC", "I-LOC",   # 地点（Location）— 在哪里
    "B-MNR", "I-MNR",   # 方式（Manner）— 怎样做
    "B-CAU", "I-CAU",   # 原因（Cause）— 为什么
    "B-PRP", "I-PRP",   # 目的（Purpose）— 为了什么
    "B-DIR", "I-DIR",   # 方向（Direction）— 向哪里
    "B-EXT", "I-EXT",   # 程度（Extent）— 到什么程度
    "V",                # 谓词（Verb）— 事件核心
    "O",                # 不属于任何角色
]


class SequenceLabelingSRL:
    """
    基于 BIO 序列标注的语义角色标注器

    ━━━━━━━ 核心思想 ━━━━━━━
    把 SRL 问题看作序列标注问题：
    - 输入：一个句子 + 谓词位置
    - 输出：每个词的 BIO 标签

    就像一条流水线上的分拣员：
    - 每个词经过流水线
    - 分拣员根据词的"特征"（词形、词性、位置等）
    - 给每个词贴上对应的角色标签

    ━━━━━━━ 特征工程 ━━━━━━━
    我们用以下特征来帮助判断标签：
    1. 当前词本身（如 "小明" 可能是人名 → 施事）
    2. 当前词的词性（如 nr=人名 → 可能是施事）
    3. 当前词与谓词的相对位置（谓词左边 vs 右边）
    4. 谓词本身是什么（如 "踢" 的施事通常是人）

    ━━━━━━━ 训练方法 ━━━━━━━
    使用平均感知机（Averaged Perceptron）：
    - 每次预测错误时，调整特征权重
    - 最终使用权重的平均值（防止过拟合）

    属性：
        weights: 特征权重字典
        averaged_weights: 平均后的权重（用于预测）
        classes: 所有可能的标签集合
    """

    def __init__(self):
        """
        初始化序列标注 SRL 模型

        ━━━━━━━ 生活类比 ━━━━━━━
        就像一个新入职的分拣员：
        - 刚开始什么都不知道（权重都是 0）
        - 需要通过"培训"（训练）来学习分拣规则
        - 培训结束后才能上岗工作（预测）
        """
        # 权重：weights[tag][feature] = 权重值
        # 例如：weights["A0"]["word=小明"] = 0.8
        # 表示"看到'小明'这个词时，它是施事的权重是 0.8"
        self.weights = _defaultdict(lambda: _defaultdict(float))

        # 平均权重：训练过程中所有权重的平均值
        # 使用平均权重可以减少过拟合，提高泛化能力
        self.averaged_weights = _defaultdict(lambda: _defaultdict(float))

        # 所有可能的标签
        self.classes = set(BIO_TAGS)

        # 训练步数计数器（用于计算平均权重）
        self._step = 0

        # 更新计数器：记录每个权重被更新了多少次
        self._update_counts = _defaultdict(lambda: _defaultdict(int))

    def _extract_features(self, sentence: list, i: int, pred_idx: int) -> dict:
        """
        提取第 i 个词的特征

        ━━━━━━━ 生活类比 ━━━━━━━
        就像分拣员看一个包裹上的信息来判断怎么分拣：
        - 包裹上写着什么（当前词）
        - 包裹的形状（词性）
        - 包裹在传送带上的位置（相对于谓词的位置）
        - 传送带上有什么特殊的标记（谓词是什么）

        ━━━━━━━ 特征说明 ━━━━━━━
        我们提取以下特征：

        1. 当前词（word=xxx）
           → 有些词本身就是角色的强信号
           → 如 "小明" → 很可能是人名 → 施事

        2. 当前词性（pos=xxx）
           → 名词更可能是论元，副词更可能是修饰语
           → 如 pos=nr（人名）→ 可能是施事

        3. 相对位置（rel_pos=xxx）
           → 谓词左边的词更可能是施事
           → 谓词右边的词更可能是受事
           → 我们把相对位置离散化为 5 档：far_left, near_left, at_pred, near_right, far_right

        4. 谓词词（pred=xxx）
           → 不同的谓词倾向于有不同的论元结构
           → 如 "吃" 的施事通常是人，"被" 后面的可能是施事

        5. 当前词与谓词的距离（dist=xxx）
           → 距离越远，越不可能是核心论元

        参数：
            sentence: 词列表，每个元素是 (词, 词性) 的元组
            i: 当前词的索引
            pred_idx: 谓词的索引

        返回：
            特征字典 {特征名: 特征值}
        """
        features = {}
        word, pos = sentence[i]

        # ─── 特征 1：当前词本身 ───
        # 直接用词作为特征，让模型学习哪些词倾向于什么角色
        features[f"w={word}"] = 1

        # ─── 特征 2：当前词性 ───
        # 词性是一个非常重要的特征
        # 例如：人名(nr)更可能是施事，时间词(t)更可能是时间角色
        features[f"pos={pos}"] = 1

        # ─── 特征 3：当前词的后缀（取最后一个字） ───
        # 后缀可以暗示词的类型
        # 例如：以"了""过"结尾的可能是时间相关
        if len(word) >= 2:
            features[f"suf={word[-1]}"] = 1

        # ─── 特征 4：相对位置（与谓词的距离） ───
        # 谓词左边 vs 右边，对判断角色非常重要
        # 我们把距离离散化，避免特征空间太大
        rel_pos = i - pred_idx
        if rel_pos < -3:
            features["rel_pos=far_left"] = 1
        elif rel_pos < 0:
            features["rel_pos=near_left"] = 1
        elif rel_pos == 0:
            features["rel_pos=at_pred"] = 1
        elif rel_pos <= 3:
            features["rel_pos=near_right"] = 1
        else:
            features["rel_pos=far_right"] = 1

        # ─── 特征 5：谓词本身 ───
        # 不同的谓词有不同的论元结构偏好
        # 例如："吃"需要施事和受事，"是"可能只需要两个论元
        pred_word = sentence[pred_idx][0]
        features[f"pred={pred_word}"] = 1

        # ─── 特征 6：距离的绝对值（离散化） ───
        # 绝对距离可以表示"核心论元通常在谓词附近"
        abs_dist = abs(rel_pos)
        if abs_dist <= 1:
            features["dist=1"] = 1
        elif abs_dist <= 3:
            features["dist=2-3"] = 1
        elif abs_dist <= 5:
            features["dist=4-5"] = 1
        else:
            features["dist=6+"] = 1

        # ─── 特征 7：当前词性 + 相对位置的组合特征 ───
        # 组合特征能捕捉更复杂的模式
        # 例如：人名在谓词左边 → 强信号是施事
        if rel_pos < 0:
            features[f"pos_left={pos}"] = 1
        elif rel_pos > 0:
            features[f"pos_right={pos}"] = 1

        # ─── 特征 8：当前词 + 谓词的组合特征 ───
        # 某些词和某些谓词的组合有特定含义
        features[f"wp={word}_{pred_word}"] = 1

        # ─── 特征 9：前一个词的词性 ───
        # 上下文信息：前一个词的词性可以帮助判断当前词的角色
        if i > 0:
            prev_pos = sentence[i - 1][1]
            features[f"prev_pos={prev_pos}"] = 1

        # ─── 特征 10：后一个词的词性 ───
        # 上下文信息：后一个词的词性也可以提供线索
        if i < len(sentence) - 1:
            next_pos = sentence[i + 1][1]
            features[f"next_pos={next_pos}"] = 1

        return features

    def _predict_one(self, sentence: list, i: int, pred_idx: int) -> str:
        """
        预测第 i 个词的标签（贪心解码）

        ━━━━━━━ 生活类比 ━━━━━━━
        就像分拣员看到一个包裹，根据经验快速判断它应该去哪个区域：
        - 看包裹上的信息（特征）
        - 在脑子里对每个可能的区域打分（权重求和）
        - 选择得分最高的区域（argmax）

        ━━━━━━━ 算法 ━━━━━━━
        对每个可能的标签，计算得分：
          score(tag) = Σ weight[tag][feature] * feature_value

        选择得分最高的标签作为预测结果。

        参数：
            sentence: 词列表
            i: 当前词索引
            pred_idx: 谓词索引

        返回：
            预测的标签
        """
        features = self._extract_features(sentence, i, pred_idx)

        best_tag = "O"   # 默认标签
        best_score = -1e10  # 负无穷

        # 对每个可能的标签计算得分
        for tag in self.classes:
            score = 0.0
            # 得分 = 所有特征的权重之和
            for feat, val in features.items():
                score += self.weights[tag].get(feat, 0.0) * val

            if score > best_score:
                best_score = score
                best_tag = tag

        return best_tag

    def train(self, labeled_data: list, epochs: int = 10):
        """
        训练模型（使用平均感知机算法）

        ━━━━━━━ 生活类比 ━━━━━━━
        就像培训一个分拣员：
        1. 让分拣员试分一批包裹（预测）
        2. 检查分对了没有（和正确答案比较）
        3. 如果分错了，告诉分拣员哪里搞错了（调整权重）
        4. 重复很多次（多个 epoch）
        5. 最终用"平均水平"作为分拣员的能力（平均权重）

        ━━━━━━━ 平均感知机算法 ━━━━━━━
        对每个训练样本：
          1. 用当前权重预测每个词的标签
          2. 如果预测错误：
             - 增加正确标签的权重（鼓励正确的特征组合）
             - 减少错误标签的权重（惩罚错误的特征组合）
          3. 累加权重用于最终平均

        ━━━━━━━ 为什么用"平均"？ ━━━━━━━
        感知机可能在最后几步对某些样本"过度修正"，
        导致权重波动很大。
        使用平均权重可以平滑这些波动，提高泛化能力。

        参数：
            labeled_data: 标注数据列表
                每个元素是 (sentence, pred_idx, labels)
                - sentence: [(词, 词性), ...]
                - pred_idx: 谓词索引
                - labels: [标签1, 标签2, ...]（每个词的正确标签）
            epochs: 训练轮数
        """
        print("  开始训练序列标注 SRL 模型...")
        print(f"  训练样本数: {len(labeled_data)}, 训练轮数: {epochs}")

        for epoch in range(epochs):
            # 打乱训练数据顺序（随机梯度下降的基本要求）
            _random.shuffle(labeled_data)

            correct = 0
            total = 0

            for sentence, pred_idx, gold_labels in labeled_data:
                # 对句子中的每个词进行预测
                for i in range(len(sentence)):
                    # 预测标签
                    predicted = self._predict_one(sentence, i, pred_idx)
                    gold = gold_labels[i]

                    total += 1
                    if predicted == gold:
                        correct += 1
                        continue

                    # ─── 预测错误，更新权重 ───
                    features = self._extract_features(sentence, i, pred_idx)

                    # 增加正确标签的权重（让模型更倾向于选正确的标签）
                    for feat, val in features.items():
                        self.weights[gold][feat] += val
                        self._update_counts[gold][feat] += 1

                    # 减少错误标签的权重（让模型更不倾向于选错误的标签）
                    for feat, val in features.items():
                        self.weights[predicted][feat] -= val
                        self._update_counts[predicted][feat] += 1

                    self._step += 1

            accuracy = correct / total if total > 0 else 0
            print(f"    Epoch {epoch + 1}/{epochs}: 准确率 = {accuracy:.4f} "
                  f"({correct}/{total})")

        # ─── 计算平均权重 ───
        # 训练结束后，把所有历史权重取平均
        # 这样可以减少训练后期的波动对最终模型的影响
        self.averaged_weights = _defaultdict(lambda: _defaultdict(float))
        for tag in self.weights:
            for feat in self.weights[tag]:
                self.averaged_weights[tag][feat] = self.weights[tag][feat]

        # 用平均权重替换原始权重
        self.weights = self.averaged_weights
        print("  训练完成！已使用平均权重。")

    def parse(self, sentence: list, predicate_idx: int) -> SRLResult:
        """
        对句子进行语义角色标注（贪心解码）

        ━━━━━━━ 生活类比 ━━━━━━━
        就像分拣员在流水线上工作：
        - 包裹一个接一个地经过
        - 分拣员对每个包裹独立判断（贪心）
        - 快速、简单，但可能不是全局最优

        ━━━━━━━ 贪心解码 vs 维特比解码 ━━━━━━━
        - 贪心：每个位置独立选最好的标签 → 快但可能不一致
        - 维特比：考虑标签之间的转移关系 → 慢但更准确
        - 这里我们用贪心解码，简单且效果已经不错

        参数：
            sentence: 词列表，每个元素是 (词, 词性)
            predicate_idx: 谓词在词列表中的索引

        返回：
            SRLResult 对象，包含标注结果
        """
        # 对每个词预测标签
        tags = []
        for i in range(len(sentence)):
            tag = self._predict_one(sentence, i, predicate_idx)
            tags.append(tag)

        # 从 BIO 标签提取论元片段
        spans = self._words_to_spans(sentence, tags)

        # 构建 SRLResult
        pred_word = sentence[predicate_idx][0]
        words = [w for w, _ in sentence]
        result = SRLResult(predicate=pred_word,
                           predicate_index=predicate_idx,
                           words=words)

        for tag, text, start, end in spans:
            if tag != "V":  # 谓词不算论元
                arg = SRLArgument(tag=tag, text=text, start=start, end=end)
                result.add_argument(arg)

        return result

    def _words_to_spans(self, sentence: list, tags: list) -> list:
        """
        将 BIO 标签序列转换为论元片段

        ━━━━━━━ 生活类比 ━━━━━━━
        就像把贴了标签的包裹按标签分组：
        - B-A0 开始，后面跟着 I-A0 → 这些包裹属于"A0 组"
        - 遇到 O 或另一个 B-xxx → 当前组结束，开始新组

        ━━━━━━━ BIO 转 Span 的规则 ━━━━━━━
        1. 遇到 B-xxx → 开始一个新的 xxx 片段
        2. 遇到 I-xxx → 如果前面是 B-xxx 或 I-xxx，延续当前片段
        3. 遇到 O → 当前片段结束（如果有）
        4. 遇到 B-xxx 但前面是 B-yyy 或 I-yyy → 前面的片段结束，开始新片段

        参数：
            sentence: 词列表
            tags: BIO 标签列表

        返回：
            片段列表 [(角色, 文本, 起始索引, 结束索引), ...]
        """
        spans = []
        current_role = None    # 当前正在构建的角色
        current_text = ""      # 当前片段的文本
        current_start = -1     # 当前片段的起始位置

        for i, tag in enumerate(tags):
            word = sentence[i][0] if isinstance(sentence[i], tuple) else sentence[i]

            if tag.startswith("B-"):
                # ─── B-xxx：开始新片段 ───
                # 先保存之前的片段（如果有的话）
                if current_role is not None:
                    spans.append((current_role, current_text,
                                  current_start, i))
                # 开始新片段
                current_role = tag[2:]   # 去掉 "B-" 前缀，得到角色名
                current_text = word
                current_start = i

            elif tag.startswith("I-"):
                # ─── I-xxx：延续当前片段 ───
                role = tag[2:]
                if current_role == role:
                    # 正确延续：追加到当前片段
                    current_text += word
                else:
                    # 不匹配：结束旧片段，开始新片段
                    if current_role is not None:
                        spans.append((current_role, current_text,
                                      current_start, i))
                    current_role = role
                    current_text = word
                    current_start = i

            elif tag == "V":
                # ─── V：谓词 ───
                # 先保存之前的片段
                if current_role is not None:
                    spans.append((current_role, current_text,
                                  current_start, i))
                    current_role = None
                    current_text = ""
                # 谓词本身也作为一个片段（方便后续处理）
                spans.append(("V", word, i, i + 1))

            else:
                # ─── O：不属于任何角色 ───
                # 结束当前片段
                if current_role is not None:
                    spans.append((current_role, current_text,
                                  current_start, i))
                    current_role = None
                    current_text = ""

        # 保存最后一个片段
        if current_role is not None:
            spans.append((current_role, current_text,
                          current_start, len(tags)))

        return spans


def demo_sequence_srl():
    """
    演示基于序列标注的 SRL

    ━━━━━━━ 演示内容 ━━━━━━━
    1. 构造训练数据
    2. 训练模型
    3. 用模型进行预测
    """
    print("=" * 60)
    print("基于 BIO 序列标注的语义角色标注")
    print("=" * 60)

    print("""
    ━━━━━━━ 序列标注的思路 ━━━━━━━

    传统规则方法：用人工定义的规则判断角色
    序列标注方法：让机器自动学习判断规则

    就像：
    - 规则方法 = 老师傅手把手教你分拣
    - 序列标注 = 你自己看大量案例，总结规律

    BIO 标注示例：
    ┌──────────────────────────────────────────────┐
    │  小明  昨天  在  学校  踢  足球                │
    │  B-A0 B-TMP B-LOC I-LOC  V  B-A1             │
    │  施事  时间  地点开始 地点延续 谓词 受事        │
    └──────────────────────────────────────────────┘
    """)

    # ─── 构造训练数据 ───
    # 每条数据：(句子, 谓词索引, 标签列表)
    # 句子格式：[(词, 词性), ...]
    training_data = [
        # 例1: 小明昨天踢了足球
        (
            [("小明", "nr"), ("昨天", "t"), ("踢", "v"), ("了", "u"), ("足球", "n")],
            2,  # 谓词 "踢" 的索引
            ["B-A0", "B-TMP", "V", "O", "B-A1"]
        ),
        # 例2: 我在学校学习自然语言处理
        (
            [("我", "r"), ("在", "p"), ("学校", "n"), ("学习", "v"), ("自然", "n"), ("语言", "n"), ("处理", "n")],
            3,  # 谓词 "学习" 的索引
            ["B-A0", "B-LOC", "I-LOC", "V", "B-A1", "I-A1", "I-A1"]
        ),
        # 例3: 他用筷子吃面条
        (
            [("他", "r"), ("用", "p"), ("筷子", "n"), ("吃", "v"), ("面条", "n")],
            3,  # 谓词 "吃" 的索引
            ["B-A0", "B-MNR", "I-MNR", "V", "B-A1"]
        ),
        # 例4: 老师昨天在教室讲了课
        (
            [("老师", "n"), ("昨天", "t"), ("在", "p"), ("教室", "n"), ("讲", "v"), ("了", "u"), ("课", "n")],
            4,  # 谓词 "讲" 的索引
            ["B-A0", "B-TMP", "B-LOC", "I-LOC", "V", "O", "B-A1"]
        ),
        # 例5: 妈妈在厨房做了晚饭
        (
            [("妈妈", "n"), ("在", "p"), ("厨房", "n"), ("做", "v"), ("了", "u"), ("晚饭", "n")],
            3,  # 谓词 "做" 的索引
            ["B-A0", "B-LOC", "I-LOC", "V", "O", "B-A1"]
        ),
        # 例6: 小红今天买了新书
        (
            [("小红", "nr"), ("今天", "t"), ("买", "v"), ("了", "u"), ("新", "a"), ("书", "n")],
            2,  # 谓词 "买" 的索引
            ["B-A0", "B-TMP", "V", "O", "B-A1", "I-A1"]
        ),
        # 例7: 他快速跑了
        (
            [("他", "r"), ("快速", "d"), ("跑", "v"), ("了", "u")],
            2,  # 谓词 "跑" 的索引
            ["B-A0", "B-MNR", "V", "O"]
        ),
        # 例8: 我们用笔写了信
        (
            [("我们", "r"), ("用", "p"), ("笔", "n"), ("写", "v"), ("了", "u"), ("信", "n")],
            3,  # 谓词 "写" 的索引
            ["B-A0", "B-MNR", "I-MNR", "V", "O", "B-A1"]
        ),
    ]

    # ─── 扩充训练数据（通过复制增加训练量） ───
    # 实际项目中应该有更多真实数据
    training_data = training_data * 5

    # ─── 训练模型 ───
    model = SequenceLabelingSRL()
    model.train(training_data, epochs=5)

    # ─── 测试预测 ───
    print("\n  测试预测:")
    print(f"  {'─' * 55}")

    test_cases = [
        # 测试句子，谓词索引
        ([("小明", "nr"), ("在", "p"), ("公园", "n"), ("踢", "v"), ("足球", "n")], 3),
        ([("我", "r"), ("昨天", "t"), ("买", "v"), ("了", "u"), ("一本书", "n")], 2),
        ([("华为", "nt"), ("在", "p"), ("北京", "ns"), ("发布", "v"), ("新手机", "n")], 3),
        ([("他", "r"), ("用", "p"), ("电脑", "n"), ("写", "v"), ("代码", "n")], 3),
    ]

    for sentence, pred_idx in test_cases:
        words_str = "".join(w for w, _ in sentence)
        pred_word = sentence[pred_idx][0]
        print(f"\n  句子: {words_str}")
        print(f"  谓词: {pred_word} (索引 {pred_idx})")

        result = model.parse(sentence, pred_idx)
        result.display()

        triple = result.to_triple()
        if triple[0] != "未知" and triple[2] != "未知":
            print(f"  三元组: {triple}")

    print("""
    ━━━━━━━ 分析 ━━━━━━━
    序列标注方法的优点：
    1. 自动学习特征权重，不需要人工定义规则
    2. 可以处理更复杂的句式
    3. 训练数据越多，效果越好

    局限性：
    1. 需要标注数据（标注成本高）
    2. 贪心解码可能不是全局最优
    3. 没有考虑标签之间的转移关系

    改进方向：
    1. 使用 CRF（条件随机场）替代贪心解码
    2. 使用深度学习（BERT）提取更好的特征
    3. 使用更大规模的标注数据
    """)
