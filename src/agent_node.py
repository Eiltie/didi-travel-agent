"""各 Agent 的节点定义。

每个节点都是一个"收 state、返回部分更新"的普通函数：
    - 入参：完整的 state（dict）
    - 返回：一个 dict，只包含自己要改的字段，其余字段 LangGraph 会自动保留
"""

import json
from datetime import date, timedelta

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_state import AgentState
from agent_tools import get_directions, get_weather, web_search
from get_llm import get_llm
from total_prompts import (
    PARSE_QUERY_PROMPT,
    PLANNER_PROMPT,
    ROUTE_PROMPT,
    ROUTE_TASK_PROMPT,
    SPOTS_PROMPT,
    SPOTS_TASK_PROMPT,
    WEATHER_SUMMARY_PROMPT,
)


# ===== 常量 =====

MAX_SEARCH = 3          # 景点 Agent 最多搜几轮，超过就不再给它工具
MAX_ROUTE_ROUNDS = 8    # 路线 Agent 最多问几轮，超过就逼它用手上的信息收工


# ===== 天气节点 =====

def weather_node(state: AgentState) -> dict:
    """天气节点：读懂需求 → 查真实天气 → 写成给人看的总结。"""
    llm = get_llm()
    user_query = state["user_query"]

    # ① 让大脑从用户原话里解析出「目的地」「天数」「出发日期」
    today = date.today()
    prompt = (PARSE_QUERY_PROMPT
              .replace("{today}", str(today))
              .replace("{weekday}", "一二三四五六日"[today.weekday()])
              .replace("{tomorrow}", str(today + timedelta(days=1)))
              .replace("{user_query}", user_query))
    raw = llm.invoke(prompt).content.strip()
    # 容错：LLM 常把 JSON 裹在 ```json ... ``` 代码块里
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        destination = (parsed.get("destination") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        destination = ""

    if not destination:
        raise ValueError(f"没能从「{user_query}」里识别出目的地。把话说具体点试试，"
                         f"比如「下周想去杭州玩3天」。")

    days = int(parsed["days"])
    start_date = parsed["start_date"]

    # ② 用手去查真实天气（直接调工具，不经过 LLM）
    weather_data = get_weather.invoke({
        "city": destination, "days": days, "start_date": start_date,
    })

    # ③ 让大脑把天气数据写成给人看的总结
    prompt = (WEATHER_SUMMARY_PROMPT
              .replace("{weather_data}", weather_data)
              .replace("{user_query}", user_query))
    summary = llm.invoke(prompt).content.strip()

    return {
        "destination": destination,
        "days": days,
        "start_date": start_date,
        "weather": summary,
    }


# ===== 景点节点 =====

def _count_searches(messages: list) -> int:
    """数一数已经搜过几次：把每条 AI 消息里的 tool_calls 加起来。"""
    return sum(len(m.tool_calls) for m in messages if getattr(m, "tool_calls", None))


def spots_agent_node(state: AgentState) -> dict:
    """景点 Agent 的"大脑"：这个节点会被反复执行，直到它说"不用再搜了"。

    每一轮它只有两种结局：
        - 回一条带 tool_calls 的消息   → "我还要搜" → 路由把它送去 tools 节点
        - 回一条不带 tool_calls 的消息 → "以上就是推荐" → 路由让它收工
    """
    history = state["messages"]

    opening = []
    if not history:
        opening = [
            SystemMessage(content=SPOTS_PROMPT),
            HumanMessage(content=SPOTS_TASK_PROMPT
                         .replace("{destination}", state["destination"])
                         .replace("{days}", str(state["days"]))
                         .replace("{start_date}", state["start_date"])
                         .replace("{weather}", state["weather"])
                         .replace("{user_query}", state["user_query"])),
        ]
        history = opening

    llm = get_llm()
    if _count_searches(history) < MAX_SEARCH:
        llm = llm.bind_tools([web_search])

    # 交给大脑思考
    ai = llm.invoke(history)

    # 交出这一轮的成果
    result = {"messages": opening + [ai]}
    if not ai.tool_calls:
        # 没调工具，说明这条就是最终清单
        result["spots"] = ai.content or "（本次没能生成景点推荐，请重试）"
    return result


# ===== 规划节点 =====

def planner_node(state: AgentState) -> dict:
    """规划节点：把景点清单分配到每一天，排出逐日行程。

    三个 Agent 里最简单的一个——它不需要工具，因为所有原料都已经在 state 里了
    （天气、景点清单、天数、同行人）。所以只做一件事：填提示词 → 调一次 LLM → 交结果。
    """
    prompt = (PLANNER_PROMPT
              .replace("{destination}", state["destination"])
              .replace("{days}", str(state["days"]))
              .replace("{start_date}", state["start_date"])
              .replace("{weather}", state["weather"])
              .replace("{spots}", state["spots"])
              .replace("{user_query}", state["user_query"]))

    itinerary = get_llm().invoke(prompt).content.strip()

    return {"itinerary": itinerary or "（本次没能生成行程安排，请重试）"}


# ===== 路线节点 =====

def route_agent_node(state: AgentState) -> dict:
    """路线节点：把逐日行程细化成带时刻和交通衔接的详细路线。

    这里是"节点内 ReAct"——循环藏在这个函数内部，图上看不见。
    所以 create_graph.py 里它就是一个普通节点，route.py 也不用动。

    每一轮问 LLM 一次，它只有两种回答：
        - 带 tool_calls  → "我要查这几段路线" → 我们替它调 get_directions，
                           把结果接在消息后面，再来一轮
        - 不带 tool_calls → "路线排好了" → 循环结束，它的正文就是最终结果
    """
    messages = [
        SystemMessage(content=ROUTE_PROMPT),
        HumanMessage(content=ROUTE_TASK_PROMPT
                     .replace("{destination}", state["destination"])
                     .replace("{days}", str(state["days"]))
                     .replace("{start_date}", state["start_date"])
                     .replace("{user_query}", state["user_query"])
                     .replace("{itinerary}", state["itinerary"])),
    ]

    llm = get_llm().bind_tools([get_directions])

    ai = None
    for _ in range(MAX_ROUTE_ROUNDS):
        ai = llm.invoke(messages)
        if not ai.tool_calls:
            break                       # 它说不用查了，正文就是路线
        messages.append(ai)
        for call in ai.tool_calls:      # 它想查的每一段，我们替它去查
            try:
                result = get_directions.invoke(call["args"])
            except Exception as e:      # 参数传错（比如漏了 city）也别让程序崩
                result = f"这个查询没成功（{e}），请检查参数后重试。"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    if ai.tool_calls:
        # 轮次用完了它还在要工具：这次不给工具，逼它用手上的信息把路线写出来
        ai = get_llm().invoke(messages)

    return {"route_plan": ai.content or "（本次没能生成路线，请重试）"}
