"""
==============================================================================
第二章：中文分词
==============================================================================
日期：2026-05-16

同学们好！这节课我们来学习中文 NLP 最基础也最重要的技术 —— 分词。

----------------------------------------------------------------------
生活类比：分词就像切蛋糕
----------------------------------------------------------------------

想象你面前有一整块长条蛋糕：

  ┌──────────────────────────────────────┐
  │  今天天气真好我们去公园玩吧            │
  └──────────────────────────────────────┘

你要把它切成小块，每块都是一个"有意义的词"：

  ┌──────┬──────┬──────┬──────┬──────┬────┐
  │ 今天 │ 天气 │ 真好 │ 我们 │ 去  │ 公园 │
  └──────┴──────┴──────┴──────┴──────┴────┘

但问题来了 —— 不同的切法会得到不同的结果：

  切法1: 今天 / 天气 / 真好
  切法2: 今 / 天天 / 气 / 真好   ← 错了！

这就是中文分词的挑战：如何找到正确的切分方式？

----------------------------------------------------------------------
为什么中文分词很难？
----------------------------------------------------------------------

例子 1："南京市长江大桥"
  - 切法1: 南京市 / 长江大桥  ← 南京市的长江大桥
  - 切法2: 南京 / 市长 / 江大桥  ← 南京的市长叫江大桥？

例子 2："下雨天留客天留我不留"
  - 切法1: 下雨天 / 留客 / 天留 / 我不留
  - 切法2: 下雨 / 天留客 / 天留我不留

例子 3："乒乓球拍卖完了"
  - 切法1: 乒乓球 / 拍卖 / 完了  ← 乒乓球的拍卖结束了
  - 切法2: 乒乓球拍 / 卖完了    ← 乒乓球拍被卖完了

这些例子说明：中文分词需要结合上下文和语义才能正确切分。

==============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ==============================================================================
# 第一部分：规则分词 —— 基于词典的分词方法
# ==============================================================================
#
# 规则分词是最简单的分词方法：
#   1. 准备一个词典（包含所有已知的词）
#   2. 按照一定的规则，从句子中匹配词典里的词
#   3. 匹配到的就切出来
#
# 就像查字典：遇到不认识的词，就去字典里找。
#
# ==============================================================================


# 默认词典
# 这个词典包含了常见的中文词汇
# 在实际项目中，词典可能有几十万甚至上百万个词
DEFAULT_DICT = {
    "今天", "天气", "真好", "我们", "去", "公园", "玩",
    "南京", "南京市", "长江", "长江大桥", "大桥", "市长",
    "自然", "语言", "处理", "自然语言", "自然语言处理",
    "机器", "学习", "机器学习", "深度", "深度学习", "人工", "智能",
    "人工智能", "中国", "中国人", "人民", "民主", "共和国",
    "中华人民共和国", "北京", "北京市", "大学", "北京大学",
    "研究", "研究生", "生命", "科学", "生命科学",
    "我", "喜欢", "吃", "苹果", "苹果公司", "手机",
    "乒乓球", "球拍", "拍卖", "乒乓球拍", "完了",
    "下雨", "天下", "下雨天", "留客", "天留", "我不",
}


def forward_max_match(text: str, dictionary: set, max_word_len: int = 5) -> list:
    """
    正向最大匹配法（Forward Maximum Matching, FMM）

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你从蛋糕的左边开始切，每次都尽量切最大的一块。
    如果最大的那块不在词典里，就切小一点，直到找到为止。

    ━━━━━━━ 算法步骤 ━━━━━━━
    1. 从句子左边开始
    2. 尝试匹配最长的词（长度为 max_word_len）
    3. 如果在词典中找到了，就切出来，继续处理剩余部分
    4. 如果没找到，把长度减 1，再试
    5. 如果长度减到 1 还没找到，就把当前这个字当作一个词

    参数：
        text: 待分词的中文句子
        dictionary: 词典（集合类型）
        max_word_len: 词典中最长词的长度，默认 5

    返回：
        分词结果列表

    示例：
        >>> forward_max_match("今天天气真好", {"今天", "天气", "真好"})
        ['今天', '天气', '真好']
    """
    result = []      # 存储分词结果
    i = 0            # 当前处理到的位置

    # 从左到右扫描整个句子
    while i < len(text):
        matched = False  # 标记是否匹配成功

        # 尝试从最长的词开始匹配
        # 比如 max_word_len=5，就依次尝试长度 5, 4, 3, 2, 1
        for length in range(max_word_len, 0, -1):
            # 如果剩余文本长度不够，跳过
            if i + length > len(text):
                continue

            # 截取当前窗口的文本
            word = text[i:i + length]

            # 检查是否在词典中
            if word in dictionary:
                result.append(word)  # 匹配成功，加入结果
                i += length          # 移动窗口
                matched = True
                break                # 跳出内层循环，继续外层循环

        # 如果所有长度都没匹配上，把当前字当作一个词
        if not matched:
            result.append(text[i])
            i += 1

    return result


def backward_max_match(text: str, dictionary: set, max_word_len: int = 5) -> list:
    """
    逆向最大匹配法（Backward Maximum Matching, BMM）

    ━━━━━━━ 生活类比 ━━━━━━━
    这次我们从蛋糕的右边开始切，也是每次都尽量切最大的一块。
    就像从右往左读书一样。

    ━━━━━━━ 算法步骤 ━━━━━━━
    1. 从句子右边开始
    2. 尝试匹配最长的词（长度为 max_word_len）
    3. 如果在词典中找到了，就切出来，继续处理左边剩余部分
    4. 如果没找到，把长度减 1，再试
    5. 如果长度减到 1 还没找到，就把当前这个字当作一个词

    参数：
        text: 待分词的中文句子
        dictionary: 词典（集合类型）
        max_word_len: 词典中最长词的长度，默认 5

    返回：
        分词结果列表（顺序是从左到右）

    示例：
        >>> backward_max_match("今天天气真好", {"今天", "天气", "真好"})
        ['今天', '天气', '真好']
    """
    result = []      # 存储分词结果（注意：从右往左收集，最后要反转）
    i = len(text)    # 当前处理到的位置（从末尾开始）

    # 从右到左扫描整个句子
    while i > 0:
        matched = False

        # 尝试从最长的词开始匹配
        for length in range(max_word_len, 0, -1):
            # 计算起始位置
            start = i - length
            if start < 0:
                continue

            # 截取当前窗口的文本
            word = text[start:i]

            # 检查是否在词典中
            if word in dictionary:
                result.append(word)  # 匹配成功，加入结果
                i = start            # 移动窗口
                matched = True
                break

        # 如果所有长度都没匹配上，把当前字当作一个词
        if not matched:
            result.append(text[i - 1])
            i -= 1

    # 因为是从右往左收集的，所以要反转
    result.reverse()
    return result


def bidirectional_max_match(text: str, dictionary: set, max_word_len: int = 5) -> list:
    """
    双向最大匹配法（Bidirectional Maximum Matching）

    ━━━━━━━ 生活类比 ━━━━━━━
    两个人同时从蛋糕的两端开始切，一个从左切，一个从右切。
    然后比较谁切得更好（词数更少、单字更少）。

    ━━━━━━━ 算法步骤 ━━━━━━━
    1. 分别用正向最大匹配和逆向最大匹配分词
    2. 比较两种结果：
       a. 如果词数不同，选择词数少的
       b. 如果词数相同，选择单字词少的
       c. 如果还相同，选择正向的结果

    参数：
        text: 待分词的中文句子
        dictionary: 词典（集合类型）
        max_word_len: 词典中最长词的长度，默认 5

    返回：
        分词结果列表
    """
    # 分别用正向和逆向分词
    forward_result = forward_max_match(text, dictionary, max_word_len)
    backward_result = backward_max_match(text, dictionary, max_word_len)

    # 比较规则 1：词数更少的更好
    if len(forward_result) != len(backward_result):
        return forward_result if len(forward_result) < len(backward_result) else backward_result

    # 比较规则 2：单字词更少的更好
    # 单字词就是长度为 1 的词，比如 "我"、"去"
    forward_single = sum(1 for w in forward_result if len(w) == 1)
    backward_single = sum(1 for w in backward_result if len(w) == 1)

    if forward_single != backward_single:
        return forward_result if forward_single < backward_single else backward_result

    # 比较规则 3：都一样就用正向的结果
    return forward_result


# ==============================================================================
# 第二部分：统计分词 —— 基于概率的分词方法
# ==============================================================================
#
# 规则分词的缺点：依赖词典，遇到新词就切不好。
# 统计分词的思路：用概率来判断哪种切分方式最可能正确。
#
# 就像天气预报：不是100%确定明天会下雨，
# 而是根据历史数据，下雨的概率是80%。
#
# ==============================================================================


# 简化的 HMM 分词模型
# HMM（隐马尔可夫模型）是一种统计模型
#
# 在分词中，我们把每个字标记为：
#   B = Begin  （词的开头）
#   M = Middle （词的中间）
#   E = End    （词的结尾）
#   S = Single （单独成词）
#
# 例如："今天天气真好"
#   标记为：B E B M E S E
#   对应：  今/天 天/气 真 好
#
# 就像给每个字贴标签，标签决定了这个词的边界在哪里。

class HMMSegmentation:
    """
    简化的 HMM 分词模型

    ━━━━━━━ 生活类比 ━━━━━━━
    想象你在看一部侦探片，你需要根据线索推断每个字是"词的开头"还是"词的结尾"。
    HMM 就是根据大量已标注的数据，学习出每个字作为 B/M/E/S 的概率。

    ━━━━━━━ 核心思想 ━━━━━━━
    1. 每个字有一个"隐藏状态"（B/M/E/S）
    2. 我们能看到字，但看不到它的状态
    3. 通过概率计算，推断最可能的状态序列
    4. 根据状态序列确定分词边界
    """

    def __init__(self):
        """初始化 HMM 模型"""

        # 状态集合
        # B=Begin, M=Middle, E=End, S=Single
        self.states = ['B', 'M', 'E', 'S']

        # 初始状态概率（每个状态作为句子开头的概率）
        # 比如：句子开头很可能是 B（词的开始）或 S（单字词）
        # 这里用简化的经验值
        self.start_prob = {
            'B': 0.5,   # 句子开头是词的开始
            'M': 0.0,   # 句子开头不可能是词的中间
            'E': 0.0,   # 句子开头不可能是词的结尾
            'S': 0.5,   # 句子开头是单字词
        }

        # 状态转移概率（从一个状态转移到另一个状态的概率）
        # 比如：B 后面最可能是 M（词还没结束）或 E（词结束了）
        self.trans_prob = {
            'B': {'B': 0.1, 'M': 0.4, 'E': 0.4, 'S': 0.1},
            'M': {'B': 0.1, 'M': 0.3, 'E': 0.5, 'S': 0.1},
            'E': {'B': 0.4, 'M': 0.0, 'E': 0.0, 'S': 0.6},
            'S': {'B': 0.4, 'M': 0.0, 'E': 0.0, 'S': 0.6},
        }

        # 发射概率（某个状态下出现某个字的概率）
        # 这里用简化版本：默认概率
        self.emit_prob = {}

    def train(self, tagged_sentences):
        """
        从已标注的句子中学习概率

        参数：
            tagged_sentences: 已标注的句子列表
                每个句子是 (字列表, 状态列表) 的元组
                例如: (['今', '天'], ['B', 'E'])
        """
        # 统计初始状态出现次数
        start_count = {s: 0 for s in self.states}
        # 统计状态转移次数
        trans_count = {s: {t: 0 for t in self.states} for s in self.states}
        # 统计发射次数
        emit_count = {s: {} for s in self.states}
        # 统计每个状态的总次数
        state_count = {s: 0 for s in self.states}

        for chars, states in tagged_sentences:
            # 统计初始状态
            start_count[states[0]] += 1

            for i in range(len(states)):
                # 统计状态次数
                state_count[states[i]] += 1
                # 统计发射次数
                char = chars[i]
                if char not in emit_count[states[i]]:
                    emit_count[states[i]][char] = 0
                emit_count[states[i]][char] += 1

                # 统计转移次数
                if i < len(states) - 1:
                    trans_count[states[i]][states[i + 1]] += 1

        # 计算概率（用拉普拉斯平滑避免零概率）
        total_starts = sum(start_count.values())
        for s in self.states:
            self.start_prob[s] = (start_count[s] + 1) / (total_starts + len(self.states))

        for s in self.states:
            total_trans = sum(trans_count[s].values())
            for t in self.states:
                self.trans_prob[s][t] = (trans_count[s][t] + 1) / (total_trans + len(self.states))

        self.emit_prob = emit_count
        self.state_count = state_count

    def _get_emit_prob(self, state, char):
        """获取发射概率，使用拉普拉斯平滑"""
        count = self.emit_prob.get(state, {}).get(char, 0)
        total = self.state_count.get(state, 1)
        vocab_size = sum(len(v) for v in self.emit_prob.values())
        return (count + 1) / (total + vocab_size)

    def viterbi(self, text: str) -> list:
        """
        维特比算法 —— 找到最可能的状态序列

        ━━━━━━━ 生活类比 ━━━━━━━
        想象你在走迷宫，每个路口都有多条路。
        你要找一条"总得分最高"的路径到达终点。
        维特比算法就是一种高效找到最优路径的方法。

        ━━━━━━━ 算法步骤 ━━━━━━━
        1. 对于句子中的每个位置，计算它处于每个状态的概率
        2. 记录到达每个状态的最优前驱
        3. 从最后一个位置回溯，得到最优状态序列

        参数：
            text: 待分词的句子

        返回：
            状态列表，如 ['B', 'E', 'B', 'M', 'E', 'S']
        """
        if not text:
            return []

        n = len(text)

        # dp[t][s] = 到达第 t 个字符、状态为 s 的最大概率
        dp = [{} for _ in range(n)]
        # path[t][s] = 到达第 t 个字符、状态为 s 的最优前驱状态
        path = [{} for _ in range(n)]

        # 初始化第一个字符
        for s in self.states:
            emit_p = self._get_emit_prob(s, text[0])
            dp[0][s] = self.start_prob[s] * emit_p
            path[0][s] = [s]

        # 递推：从第 2 个字符开始
        for t in range(1, n):
            new_path = {}
            for s in self.states:
                # 找到使 dp[t-1][prev] * trans_prob[prev][s] 最大的 prev
                max_prob = 0
                best_prev = 'S'
                for prev in self.states:
                    prob = dp[t - 1].get(prev, 0) * self.trans_prob[prev][s]
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev

                # 乘以发射概率
                emit_p = self._get_emit_prob(s, text[t])
                dp[t][s] = max_prob * emit_p
                new_path[s] = path[t - 1][best_prev] + [s]

            path[t] = new_path

        # 找到最后一个字符的最优状态
        best_last_state = max(dp[n - 1], key=dp[n - 1].get)
        return path[n - 1][best_last_state]

    def segment(self, text: str) -> list:
        """
        使用 HMM 模型对文本进行分词

        参数：
            text: 待分词的句子

        返回：
            分词结果列表
        """
        states = self.viterbi(text)

        result = []
        word = ""
        for i, (char, state) in enumerate(zip(text, states)):
            if state == 'B':
                # 词的开始
                word = char
            elif state == 'M':
                # 词的中间
                word += char
            elif state == 'E':
                # 词的结束
                word += char
                result.append(word)
                word = ""
            elif state == 'S':
                # 单独成词
                result.append(char)

        # 如果最后还有未完成的词
        if word:
            result.append(word)

        return result


# ==============================================================================
# 第三部分：jieba 分词实战
# ==============================================================================
#
# jieba 是 Python 中最流行的中文分词工具。
# 它结合了规则分词和统计分词的优点：
#   1. 先用词典匹配（规则分词）
#   2. 对于词典中没有的词，用 HMM 识别新词（统计分词）
#
# 就像一个经验丰富的老师傅：
#   - 常见的词他一眼就认出来（词典匹配）
#   - 遇到没见过的词，他根据经验推测（HMM 模型）
#
# ==============================================================================

def demo_jieba_segmentation():
    """
    演示 jieba 分词的各种模式

    jieba 有三种分词模式：
    1. 精确模式 —— 最常用的模式，尽量精确地切分
    2. 全模式 —— 把所有可能的词都切出来（速度快但不精确）
    3. 搜索引擎模式 —— 在精确模式基础上，对长词再切分（适合搜索）
    """
    print("=" * 60)
    print("jieba 分词实战")
    print("=" * 60)

    try:
        import jieba
        import jieba.posseg as pseg
    except ImportError:
        print("\n[提示] jieba 未安装，请运行: pip install jieba")
        print("下面用伪代码演示 jieba 的使用方式：")
        print("""
        import jieba

        # 精确模式（最常用）
        words = jieba.cut("我来到北京清华大学", cut_all=False)
        # 结果: ['我', '来到', '北京', '清华大学']

        # 全模式（所有可能的词）
        words = jieba.cut("我来到北京清华大学", cut_all=True)
        # 结果: ['我', '来到', '北京', '清华', '清华大学', '华大', '大学']

        # 搜索引擎模式（适合搜索）
        words = jieba.cut_for_search("我来到北京清华大学")
        # 结果: ['我', '来到', '北京', '清华', '华大', '大学', '清华大学']
        """)
        return

    test_text = "我来到北京清华大学学习自然语言处理"
    print(f"\n测试文本: {test_text}")

    # ---------------------------------------------------------------
    # 精确模式
    # ---------------------------------------------------------------
    # cut_all=False 表示精确模式
    # 它会尽量精确地切分，适合文本分析
    print("\n--- 精确模式（cut_all=False）---")
    words精确 = jieba.cut(test_text, cut_all=False)
    print(f"结果: {' / '.join(words精确)}")

    # ---------------------------------------------------------------
    # 全模式
    # ---------------------------------------------------------------
    # cut_all=True 表示全模式
    # 它会把所有可能的词都切出来，速度快但可能有歧义
    print("\n--- 全模式（cut_all=True）---")
    words全 = jieba.cut(test_text, cut_all=True)
    print(f"结果: {' / '.join(words全)}")

    # ---------------------------------------------------------------
    # 搜索引擎模式
    # ---------------------------------------------------------------
    # cut_for_search 在精确模式基础上，对长词再切分
    # 适合搜索引擎使用
    print("\n--- 搜索引擎模式 ---")
    words搜索 = jieba.cut_for_search(test_text)
    print(f"结果: {' / '.join(words搜索)}")

    # ---------------------------------------------------------------
    # 词性标注
    # ---------------------------------------------------------------
    # jieba 还支持词性标注（下一章会详细讲）
    print("\n--- 词性标注 ---")
    words_pos = pseg.cut(test_text)
    for word, flag in words_pos:
        print(f"  {word} → {flag}")

    # ---------------------------------------------------------------
    # 添加自定义词典
    # ---------------------------------------------------------------
    # 如果 jieba 切错了，可以添加自定义词典
    print("\n--- 添加自定义词典 ---")
    print("默认切分: ", " / ".join(jieba.cut("赵本山今年上春晚")))

    # 添加自定义词
    jieba.add_word("赵本山", freq=10000, tag="nr")
    jieba.add_word("上春晚", freq=10000, tag="v")

    print("自定义后: ", " / ".join(jieba.cut("赵本山今年上春晚")))


# ==============================================================================
# 第四部分：分词效果评估
# ==============================================================================
#
# 如何评估分词的好坏？常用的指标有三个：
#
#   1. 准确率（Precision）—— 切出来的词中，有多少是正确的？
#      正确的词数 / 切出来的总词数
#
#   2. 召回率（Recall）—— 所有正确的词中，有多少被切出来了？
#      切出来的正确词数 / 标准答案中的总词数
#
#   3. F1 值 —— 准确率和召回率的调和平均
#      F1 = 2 * P * R / (P + R)
#
# 生活类比：
#   - 准确率 = 你钓到的鱼中，有多少是你要的鱼？
#   - 召回率 = 所有你要的鱼中，你钓到了多少？
#   - F1 = 综合评分
#
# ==============================================================================

def evaluate_segmentation(predicted: list, golden: list) -> dict:
    """
    评估分词效果

    参数：
        predicted: 模型切分的结果
            例如: ['今天', '天气', '真好']
        golden: 标准答案（人工标注的结果）
            例如: ['今天', '天气', '真', '好']

    返回：
        包含准确率、召回率、F1 值的字典
    """
    # 将分词结果转换为集合（便于计算交集）
    pred_set = set(predicted)
    gold_set = set(golden)

    # 计算交集（正确的词）
    correct = pred_set & gold_set

    # 准确率 = 正确词数 / 预测词数
    precision = len(correct) / len(pred_set) if pred_set else 0

    # 召回率 = 正确词数 / 标准答案词数
    recall = len(correct) / len(gold_set) if gold_set else 0

    # F1 值 = 2 * P * R / (P + R)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "correct_count": len(correct),
        "pred_count": len(pred_set),
        "gold_count": len(gold_set),
    }


# ==============================================================================
# 演示函数
# ==============================================================================

def demo_rule_based_segmentation():
    """演示规则分词的三种方法"""

    print("=" * 60)
    print("规则分词演示")
    print("=" * 60)

    test_cases = [
        ("今天天气真好", {"今天", "天气", "真好"}),
        ("南京市有长江大桥", {"南京市", "有", "长江大桥", "南京", "长江", "大桥", "市长"}),
        ("我喜欢自然语言处理", {"我", "喜欢", "自然语言处理", "自然", "语言", "处理", "自然语言"}),
    ]

    for text, dictionary in test_cases:
        print(f"\n句子: {text}")
        print(f"词典: {dictionary}")

        fmm = forward_max_match(text, dictionary)
        bmm = backward_max_match(text, dictionary)
        bi = bidirectional_max_match(text, dictionary)

        print(f"  正向最大匹配: {fmm}")
        print(f"  逆向最大匹配: {bmm}")
        print(f"  双向最大匹配: {bi}")


def demo_hmm_segmentation():
    """演示 HMM 分词"""

    print("\n" + "=" * 60)
    print("HMM 统计分词演示")
    print("=" * 60)

    # 创建 HMM 模型
    hmm = HMMSegmentation()

    # 用简单的训练数据
    # 格式: (字列表, 状态列表)
    training_data = [
        (['今', '天', '天', '气', '真', '好'], ['B', 'E', 'B', 'E', 'S', 'S']),
        (['我', '喜', '欢', '学', '习'], ['S', 'B', 'E', 'B', 'E']),
        (['自', '然', '语', '言', '处', '理'], ['B', 'E', 'B', 'E', 'B', 'E']),
        (['深', '度', '学', '习', '很', '有', '趣'], ['B', 'E', 'B', 'E', 'S', 'B', 'E']),
    ]

    # 训练模型
    hmm.train(training_data)
    print("\n模型训练完成！")

    # 测试分词
    test_texts = ["今天天气好", "我喜欢学习", "深度学习有趣"]

    for text in test_texts:
        result = hmm.segment(text)
        print(f"\n  '{text}' → {result}")


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

    # 运行所有演示
    demo_rule_based_segmentation()
    demo_hmm_segmentation()
    demo_jieba_segmentation()

    # 课程总结
    print("\n" + "=" * 60)
    print("第二章 总结")
    print("=" * 60)
    print("""
    本章我们学习了中文分词的三种方法：

    [OK] 规则分词（基于词典）
         - 正向最大匹配（FMM）：从左到右，每次切最大的词
         - 逆向最大匹配（BMM）：从右到左，每次切最大的词
         - 双向最大匹配：结合两种方法，选择更好的结果
         优点：简单快速
         缺点：依赖词典，遇到新词无能为力

    [OK] 统计分词（基于概率）
         - HMM 模型：用 B/M/E/S 标记每个字的状态
         - 维特比算法：找到最可能的状态序列
         优点：能识别新词
         缺点：需要大量训练数据

    [OK] jieba 分词实战
         - 精确模式：最常用，切分精确
         - 全模式：所有可能的词，速度快
         - 搜索引擎模式：适合搜索场景
         - 自定义词典：解决切分错误

    [OK] 分词评估
         - 准确率（Precision）
         - 召回率（Recall）
         - F1 值
    """)

    # 下节预告
    print("-" * 60)
    print("下节预告：第三章 — 词性标注")
    print("-" * 60)
    print("""
    下一章我们将学习词性标注：
    - 什么是词性？（名词、动词、形容词...）
    - HMM 词性标注的原理
    - jieba 词性标注实战

    预习建议：
    - 了解常见的中文词性分类
    - 思考："他在银行存钱"中"银行"是什么词性？
    """)
