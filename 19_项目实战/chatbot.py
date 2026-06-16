"""
==============================================================================
第十九章：项目实战 — 聊天机器人
==============================================================================
日期：2026-05-16

同学们好！这个模块我们来实现一个"基于规则的聊天机器人"。

----------------------------------------------------------------------
生活类比：聊天机器人就像"自动问答机"
----------------------------------------------------------------------

想象你在银行大厅，有一个"自动问答机"：
  你问："怎么查余额？"
  机器回答："请插入银行卡，输入密码，选择'查询余额'。"

这个机器人的工作原理：
  1. 预先存储了很多"问题-答案"对
  2. 你问一个问题
  3. 机器在数据库里找最匹配的问题
  4. 返回对应的答案

这就是"基于规则的聊天机器人"的原理。

----------------------------------------------------------------------
本模块内容
----------------------------------------------------------------------

1. 模式匹配（Pattern Matching）
2. 意图识别（Intent Recognition）
3. 实体提取（Entity Extraction）
4. 对话管理（Dialogue Management）
5. 完整的聊天机器人实现

==============================================================================
"""

import re
import random
from collections import defaultdict


# ==============================================================================
# 第一部分：模式匹配引擎
# ==============================================================================
#
# 模式匹配是聊天机器人最基础的技术。
#
# 核心思想：
#   用正则表达式或关键词来匹配用户的输入，
#   然后返回对应的回复。
#
# 生活类比：
#   想象你是一个客服：
#   - 用户说"怎么退货" → 你知道要回答退货流程
#   - 用户说"多少钱" → 你知道要回答价格信息
#   - 你在脑子里有一个"关键词 → 回答"的映射表
#
# ==============================================================================


class PatternMatcher:
    """
    模式匹配引擎

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一个"关键词识别器"：
    - 你说"天气" → 我知道你要问天气
    - 你说"你好" → 我知道你要打招呼
    """

    def __init__(self):
        """初始化模式匹配器"""
        # 存储模式和对应的回复
        # 格式：[(正则表达式, 回复列表)]
        self.patterns = []

    def add_pattern(self, pattern: str, responses: list):
        """
        添加一个匹配模式

        参数：
            pattern: 正则表达式模式
            responses: 匹配后的回复列表（随机选择一个）
        """
        self.patterns.append((re.compile(pattern), responses))

    def match(self, text: str) -> str:
        """
        匹配用户输入，返回回复

        参数：
            text: 用户输入的文本

        返回：
            匹配的回复，如果没有匹配则返回 None
        """
        for pattern, responses in self.patterns:
            if pattern.search(text):
                return random.choice(responses)
        return None


# ==============================================================================
# 第二部分：意图识别
# ==============================================================================
#
# 意图识别就是判断用户"想做什么"。
#
# 生活类比：
#   想象你是一个秘书，老板说了一句话，你要理解他的"意图"：
#   - "帮我倒杯水" → 意图：请求帮助
#   - "今天几号" → 意图：询问信息
#   - "取消会议" → 意图：执行命令
#
# ==============================================================================


class IntentClassifier:
    """
    基于关键词的意图分类器

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一个"听话听音"的助手：
    - 听到"怎么、如何、什么" → 疑问意图
    - 听到"谢谢、感谢" → 感谢意图
    - 听到"再见、拜拜" → 告别意图
    """

    def __init__(self):
        """初始化意图分类器"""
        # 意图 → 关键词列表
        self.intent_keywords = {
            "greeting": ["你好", "hi", "hello", "嗨", "早上好", "下午好", "晚上好"],
            "farewell": ["再见", "拜拜", "bye", "回头见", "下次见"],
            "thanks": ["谢谢", "感谢", "多谢", "thanks", "thank"],
            "query_weather": ["天气", "气温", "下雨", "晴天", "温度"],
            "query_time": ["几点", "时间", "日期", "今天几号"],
            "query_price": ["多少钱", "价格", "费用", "收费"],
            "help": ["帮助", "怎么用", "功能", "能做什么"],
            "complaint": ["投诉", "不满", "差评", "太差"],
        }

    def classify(self, text: str) -> str:
        """
        识别用户意图

        参数：
            text: 用户输入

        返回：
            意图标签
        """
        text_lower = text.lower()

        # 统计每个意图的匹配关键词数
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score

        if scores:
            # 返回得分最高的意图
            return max(scores, key=scores.get)

        return "unknown"


# ==============================================================================
# 第三部分：实体提取
# ==============================================================================
#
# 实体提取就是从用户输入中提取"关键信息"。
#
# 生活类比：
#   想象你是一个快递员，客户说：
#   "请把这个包裹寄到北京市海淀区中关村大街1号，收件人张三。"
#
#   你需要提取：
#   - 地址：北京市海淀区中关村大街1号
#   - 收件人：张三
#
# ==============================================================================


class EntityExtractor:
    """
    简单的实体提取器

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一个"信息提取器"：
    - 从一句话中提取关键信息
    - 比如人名、地点、时间、数字等
    """

    def __init__(self):
        """初始化实体提取器"""
        # 一些简单的实体模式
        self.patterns = {
            "phone": re.compile(r'1[3-9]\d{9}'),
            "date": re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'),
            "time": re.compile(r'\d{1,2}:\d{2}'),
            "number": re.compile(r'\d+'),
            "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
        }

    def extract(self, text: str) -> dict:
        """
        从文本中提取实体

        参数：
            text: 输入文本

        返回：
            实体字典 {实体类型: 实体值列表}
        """
        entities = {}
        for entity_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = matches
        return entities


# ==============================================================================
# 第四部分：对话管理器
# ==============================================================================
#
# 对话管理器负责维护对话的"上下文"和"状态"。
#
# 生活类比：
#   想象你是一个客服，正在和客户通电话：
#   - 你需要记住之前说了什么（上下文）
#   - 你需要知道现在聊到哪一步了（状态）
#   - 你需要根据上下文决定下一步说什么
#
# ==============================================================================


class DialogueManager:
    """
    对话管理器

    ━━━━━━━ 生活类比 ━━━━━━━
    就像一个"有记忆力的客服"：
    - 记住之前说了什么（对话历史）
    - 知道现在在聊什么（当前状态）
    - 根据上下文决定下一步说什么
    """

    def __init__(self):
        """初始化对话管理器"""
        # 对话历史
        self.history = []
        # 当前状态
        self.state = "idle"
        # 用户信息
        self.user_info = {}

    def add_to_history(self, role: str, text: str):
        """
        添加对话历史

        参数：
            role: 角色（"user" 或 "bot"）
            text: 对话内容
        """
        self.history.append({"role": role, "text": text})

    def get_context(self, n_turns: int = 3) -> list:
        """
        获取最近的对话上下文

        参数：
            n_turns: 获取最近几轮对话

        返回：
            对话历史列表
        """
        return self.history[-n_turns:]


# ==============================================================================
# 第五部分：完整的聊天机器人
# ==============================================================================


class ChatBot:
    """
    基于规则的聊天机器人

    ━━━━━━━ 生活类比 ━━━━━━━
    聊天机器人就像一个"多功能助手"：
    1. 模式匹配：听懂你在说什么
    2. 意图识别：理解你想做什么
    3. 实体提取：提取关键信息
    4. 对话管理：记住上下文
    5. 回复生成：给出合适的回答
    """

    def __init__(self, name: str = "小助手"):
        """
        初始化聊天机器人

        参数：
            name: 机器人的名字
        """
        self.name = name
        self.matcher = PatternMatcher()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.dialogue_manager = DialogueManager()

        # 初始化默认规则
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认的匹配规则"""

        # 打招呼
        self.matcher.add_pattern(
            r'你好|hi|hello|嗨',
            [
                f"你好！我是{self.name}，有什么可以帮你的吗？",
                f"嗨！很高兴见到你！我是{self.name}。",
                f"你好呀！有什么问题尽管问我！",
            ]
        )

        # 告别
        self.matcher.add_pattern(
            r'再见|拜拜|bye',
            [
                "再见！祝你一切顺利！",
                "拜拜！有需要随时找我！",
                "再见！期待下次聊天！",
            ]
        )

        # 感谢
        self.matcher.add_pattern(
            r'谢谢|感谢|多谢',
            [
                "不客气！能帮到你我很开心！",
                "不用谢！这是我应该做的！",
                "随时为你服务！",
            ]
        )

        # 询问天气
        self.matcher.add_pattern(
            r'天气|气温|下雨',
            [
                "我暂时无法查询实时天气，建议你查看天气预报APP。",
                "抱歉，我还没有接入天气数据，不过今天看起来不错！",
            ]
        )

        # 询问时间
        self.matcher.add_pattern(
            r'几点|时间|日期',
            [
                "我暂时无法获取当前时间，请看你的设备时钟。",
                "时间过得真快！不过我没法告诉你具体时间。",
            ]
        )

        # 询问功能
        self.matcher.add_pattern(
            r'帮助|怎么用|功能|能做什么',
            [
                f"我是{self.name}，可以和你聊天、回答问题！\n"
                "  我能做的：\n"
                "  - 聊天解闷\n"
                "  - 回答简单问题\n"
                "  - 提供基本帮助\n"
                "  试试和我聊聊吧！",
            ]
        )

        # 默认回复
        self.matcher.add_pattern(
            r'.*',
            [
                "我不太明白你的意思，能换个说法吗？",
                "这个问题有点难，我还在学习中...",
                "你可以问我一些简单的问题哦！",
            ]
        )

    def chat(self, user_input: str) -> str:
        """
        处理用户输入，返回回复

        ━━━━━━━ 流程 ━━━━━━━
        1. 记录用户输入
        2. 识别意图
        3. 提取实体
        4. 模式匹配获取回复
        5. 记录回复
        6. 返回回复

        参数：
            user_input: 用户输入的文本

        返回：
            机器人的回复
        """
        # 记录用户输入
        self.dialogue_manager.add_to_history("user", user_input)

        # 识别意图
        intent = self.intent_classifier.classify(user_input)

        # 提取实体
        entities = self.entity_extractor.extract(user_input)

        # 模式匹配获取回复
        response = self.matcher.match(user_input)

        # 如果没有匹配到，使用默认回复
        if response is None:
            response = "我不太明白，能再说详细一点吗？"

        # 记录回复
        self.dialogue_manager.add_to_history("bot", response)

        return response

    def get_intent(self, text: str) -> str:
        """获取用户意图"""
        return self.intent_classifier.classify(text)

    def get_entities(self, text: str) -> dict:
        """获取实体"""
        return self.entity_extractor.extract(text)


# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        G-one NLP 学院 - 第十九章                      ║
    ║        项目实战：聊天机器人                           ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # 创建聊天机器人
    bot = ChatBot(name="小智")

    # 测试对话
    test_inputs = [
        "你好",
        "今天天气怎么样？",
        "现在几点了？",
        "你能做什么？",
        "谢谢你的帮助",
        "再见",
    ]

    print("=" * 60)
    print("  聊天机器人演示")
    print("=" * 60)

    for user_input in test_inputs:
        response = bot.chat(user_input)
        intent = bot.get_intent(user_input)
        entities = bot.get_entities(user_input)

        print(f"\n  用户: {user_input}")
        print(f"  意图: {intent}")
        if entities:
            print(f"  实体: {entities}")
        print(f"  机器人: {response}")

    # 实体提取演示
    print("\n" + "=" * 60)
    print("  实体提取演示")
    print("=" * 60)

    test_texts = [
        "我的电话是13812345678",
        "请在2024-01-15之前发货",
        "价格是99元",
        "我的邮箱是test@example.com",
    ]

    extractor = EntityExtractor()
    for text in test_texts:
        entities = extractor.extract(text)
        print(f"  '{text}'")
        print(f"    实体: {entities}")

    # =============================================
    # 课程总结
    # =============================================
    """
    核心收获：
    - 聊天机器人的四大支柱 —— 模式匹配、意图识别、实体提取、对话管理缺一不可
    - 模式匹配用正则表达式实现关键词触发 —— 简单直接但覆盖面有限
    - 意图识别和实体提取是理解用户输入的两个维度 —— 意图回答"想做什么"，实体回答"对谁做"

    常见陷阱：
    - 默认回复放在非最后位置 —— 正则表达式是顺序匹配的，通配符 .* 必须放在最后否则会拦截所有输入
    - 过度依赖关键词匹配 —— 用户表达方式多样，仅靠关键词容易漏匹配，需要结合意图分类兜底
    - 忽略对话上下文 —— 不记录历史就无法处理指代消解（如"它多少钱"中的"它"）
    """
