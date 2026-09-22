"""评测集：给这套系统做体检——它检查的不是"能不能跑"，是"跑得对不对"。

为什么要有它：手工跑几遍，只能得出"看着没问题"；说不出好在哪、差在哪。
有了它，以后任何一个改动（换提示词、换模型、加节点）都能立刻量出来：
    改之前通过几条 → 改之后通过几条。
这就是简历上"通过率从 X% 提升到 Y%"里那两个数字的来源。

它是个【外挂】工具，项目里其他文件一个字都不用改：
    understand_query 是从 main.py 借的，build_graph 是从 create_graph.py 借的，
    用法和 api.py 里一模一样，没有复制任何逻辑。

怎么跑（在项目根目录下）：

    venv\\Scripts\\python.exe src/evals.py          ← 默认：快测全跑 + 慢测跑前 3 条
    venv\\Scripts\\python.exe src/evals.py --fast   ← 只跑快测（几十秒）
    venv\\Scripts\\python.exe src/evals.py --full   ← 慢测全跑（每条约 1.5 分钟，先算好时间）

两层评测：

    第一层 快测：只测"入口判断"——这一句是闲聊还是出行需求？不跑图，秒级。
                 这里是错误成本最高的地方：判错了，要么用户啥也得不到，
                 要么白跑一分半、还可能把程序搞崩。
    第二层 慢测：真跑整张图，看四段结果齐不齐、有没有兜底文案、每段各花多久。

快测的判定标准（这条以后改提示词时就是依据）：
    需要真实数据的 → TRAVEL；靠常识就能答的 → CHAT；信息太少、说不清要去哪的 → CHAT（先追问）。
    比如"杭州明天天气怎么样"需要真实天气数据 → TRAVEL；
        "西湖和灵隐寺哪个更值得去"常识就能答 → CHAT；
        "想去个凉快的地方待几天"没目的地 → CHAT，但要追问得好（见下）。

有一组是【人眼项】，不计入自动判定：标签为「不全」的那几条，判定一定是 CHAT，
真正要看的是回复里有没有接住用户已经说过的条件——他说"凉快"，
回复里就该出现几个凉快的地方让他挑，而不是干巴巴回一句"想去哪玩几天"。
"""

import sys
import time

from create_graph import build_graph
from main import understand_query


# ===== 用例数据 =====

FAST_CASES = [
    # ——— 正常出行需求（期望 TRAVEL）———
    ("正常", "下周想去杭州玩3天，带老人", "TRAVEL"),
    ("正常", "国庆想去成都吃4天", "TRAVEL"),
    ("正常", "帮我安排一下西安两日游", "TRAVEL"),
    ("正常", "想去青岛看海，三天，预算不多", "TRAVEL"),
    ("正常", "春节去哈尔滨玩5天，带小孩", "TRAVEL"),

    ("不全", "想去个凉快的地方待几天", "CHAT"),
    ("不全", "周末想出去走走，你看着安排", "CHAT"),
    ("不全", "有什么适合带老人玩的地方吗", "CHAT"),

    # ——— 闲聊（期望 CHAT）———
    ("闲聊", "你好", "CHAT"),
    ("闲聊", "你是谁？能干什么", "CHAT"),
    ("闲聊", "今天心情不太好", "CHAT"),

    ("天气", "杭州明天天气怎么样", "TRAVEL"),
    ("天气", "北京现在多少度", "TRAVEL"),
    ("天气", "上海这周末下雨吗", "TRAVEL"),

    # ——— 边界：靠常识就能答的（期望 CHAT，不该白跑一张图）———
    ("边界", "西湖和灵隐寺哪个更值得去", "CHAT"),
    ("边界", "杭州有什么好吃的", "CHAT"),
    ("边界", "谢谢，不用了", "CHAT"),

    # ——— 多轮：带上历史再问一句（期望 TRAVEL，且改写后要认得出上下文）———
    ("多轮", "那吃的呢？", "TRAVEL", ["下周想去杭州玩3天"]),
    ("多轮", "换个城市，去成都", "TRAVEL", ["下周想去杭州玩3天"]),
]


E2E_CASES = [
    {"tag": "标准路径 · 杭州3天", "query": "下周想去杭州玩3天，带老人",
     "expect": "ok", "keyword": "杭州", "default": True},

    {"tag": "带约束 · 成都4天带小孩", "query": "国庆想去成都玩4天，带小孩",
     "expect": "ok", "keyword": "成都", "default": True},

    {"tag": "不存在的城市", "query": "帮我查下阿斯嘉特玩3天合适吗",
     "expect": "graceful", "keyword": None, "default": True},

    {"tag": "国外城市 · 东京（已知：中文译名常查不到）", "query": "想去东京玩5天",
     "expect": "graceful", "keyword": None, "default": False},
]


# 和 api.py 里那四个字段一样：这四段齐全，才算一次完整的产出。
SECTION_KEYS = ("weather", "spots", "itinerary", "route_plan")


FALLBACK_MARKS = ("本次没能", "没跑通")


# ===== 第一层：快测（不跑图）=====

def run_fast() -> list[str]:
    """逐条问 understand_query：判成 CHAT 还是 TRAVEL。返回失败清单。"""
    print("=" * 62)
    print(f"第一层 · 快测（入口判断，不跑图）  共 {len(FAST_CASES)} 条")
    print("=" * 62)

    passed = 0
    failures = []
    t0 = time.time()

    for case in FAST_CASES:
        label, query, expected = case[0], case[1], case[2]
        history = case[3] if len(case) > 3 else []       # 第 4 个元素可有可无

        try:
            kind, content = understand_query(history, query)
        except Exception as e:
            # 连"判断"这一步都能抛异常，这也是要记录在案的问题
            kind, content = "抛异常", str(e)

        ok = (kind == expected)

        print(f"{'✅' if ok else '❌'} [{label}] {query}")
        if kind == "TRAVEL":
            # 改写后的那句话打出来——多轮用例对不对，看这一行就够
            print(f"      改写后 → {content[:64]}")
        elif label == "不全":
            print(f"      追问 → {content[:64]}")

        if ok:
            passed += 1
        else:
            failures.append(f"[{label}] {query}  —— 期望 {expected}，实际 {kind}")
            print(f"      ↑ 期望 {expected}，实际 {kind}")

    print("-" * 62)
    print(f"快测：{passed}/{len(FAST_CASES)} 通过   耗时 {time.time() - t0:.0f} 秒")
    return failures


# ===== 第二层：慢测（真跑整张图）=====

def run_one(graph, case: dict) -> str | None:
    """跑一条完整的图。通过就返回 None，不通过就返回一句失败原因。"""
    print(f"\n▶ {case['tag']}：{case['query']}")

    result = {}          # 攒四段产出（和 api.py 里那个循环一个套路）
    timings = {}         # 每一段是第几秒跑完的
    error = None
    t0 = time.time()

    try:
        for chunk in graph.stream({
            "user_query": case["query"],
            "destination": "",
            "days": 0,
            "start_date": "",
            "weather": "",
            "messages": [],      # 景点 Agent 的 ReAct"草稿纸"，开场是空的
            "spots": "",
            "itinerary": "",
            "route_plan": "",
        }):
            for _, update in chunk.items():
                result.update(update)
                for key in SECTION_KEYS:
                    # 第一次见到这个字段，就是它跑完的时刻
                    if key in update and key not in timings:
                        timings[key] = time.time() - t0
    except Exception as e:
        error = str(e)

    total = time.time() - t0

    # ——— 先把这一条的实况打出来（人眼扫一遍，比任何自动断言都实在）———
    if error:
        print(f"  跑图中断（第 {total:.0f} 秒）：{error}")
    else:
        parts = "  ".join(f"{k} {timings.get(k, 0):.0f}s" for k in SECTION_KEYS)
        print(f"  耗时：{parts}   总 {total:.0f} 秒")
        for key in SECTION_KEYS:
            text = (result.get(key) or "").strip()
            head = text[:34].replace("\n", " ⏎ ") if text else "（空）"
            print(f"    {key:<11} {len(text):>5} 字  {head}")

    # ——— 再判定 ———
    if case["expect"] == "graceful":
        # 这类用例允许失败，但失败得"体面"：报错要是给人看的中文，
        # 不能甩出 Python 堆栈或英文校验错误。
        if not error:
            return None                      # 居然跑通了，那更好
        has_chinese = any("一" <= ch <= "鿿" for ch in error)
        if has_chinese and "Traceback" not in error:
            return None
        return f"报错不体面（不是给人看的）：{error[:80]}"

    # expect == "ok"：四段必须齐，而且不能是兜底文案
    if error:
        return f"不该出错却出错了：{error[:80]}"

    for key in SECTION_KEYS:
        text = (result.get(key) or "").strip()
        if not text:
            return f"{key} 是空的"
        for mark in FALLBACK_MARKS:
            if mark in text:
                return f"{key} 是兜底文案（出现了「{mark}」）"

    keyword = case.get("keyword")
    if keyword:
        joined = "".join(result.get(k) or "" for k in SECTION_KEYS)
        if keyword not in joined:
            return f"四段里都没提到「{keyword}」"

    return None


def run_slow(full: bool) -> list[str]:
    """跑第二层。不通过就返回一句失败原因。"""
    cases = E2E_CASES if full else [c for c in E2E_CASES if c.get("default")]

    print()
    print("=" * 62)
    print(f"第二层 · 慢测（真跑整张图）  共 {len(cases)} 条"
          f"{'  ← --full 全跑' if full else '  ← 默认只跑前几条，--full 可全跑'}")
    print("=" * 62)

    graph = build_graph()      # 只编译一次，和 api.py 里一个道理
    passed = 0
    failures = []
    t0 = time.time()

    for case in cases:
        problem = run_one(graph, case)
        if problem:
            failures.append(f"{case['tag']}：{problem}")
            print(f"  ❌ {problem}")
        else:
            passed += 1
            print("  ✅ 通过")

    print("-" * 62)
    print(f"慢测：{passed}/{len(cases)} 通过   总耗时 {time.time() - t0:.0f} 秒")
    return failures


# ===== 入口 =====

if __name__ == "__main__":

    args = sys.argv[1:]
    only_fast = "--fast" in args
    full = "--full" in args

    fast_failures = run_fast()

    slow_failures = []
    if not only_fast:
        slow_failures = run_slow(full)

    # ——— 总账 ———
    print()
    print("=" * 62)
    print("总账")
    print("=" * 62)
    print(f"快测：{len(FAST_CASES) - len(fast_failures)}/{len(FAST_CASES)} 通过")
    if not only_fast:
        ran = len(E2E_CASES) if full else sum(1 for c in E2E_CASES if c.get("default"))
        print(f"慢测：{ran - len(slow_failures)}/{ran} 通过")

    for name, items in (("快测失败", fast_failures), ("慢测失败", slow_failures)):
        if items:
            print(f"\n{name}：")
            for i, line in enumerate(items, 1):
                print(f"  {i}. {line}")

    print()
    print("提醒：这只是「体检结果」，先别急着改代码——")
    print("      把这几条失败记下来，它就是你改进前的基线（那个 X）。")
