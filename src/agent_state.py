from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """全图共享状态：节点从上游字段读数据，往自己的产出字段写数据。"""

    # ===== 输入区 =====
    user_query: str      # 用户原始需求，如"下周想去杭州玩3天，带老人"
    destination: str     # 目的地（由天气 Agent 从 user_query 中解析后写入，供后续 Agent 复用）
    days: int            # 需要查询的天数（由天气 Agent 解析，默认 3）
    start_date: str      # 出发日期 YYYY-MM-DD（由天气 Agent 解析，供后续 Agent 复用）

    # ===== 天气 Agent 产出 =====
    weather: str         # 天气查询结果（自然语言文本，供后续 Agent 阅读）

    # ===== 景点 Agent 产出 =====
    # messages 是 ReAct 循环的"草稿纸"：整条图执行期间，每轮都要往上"追加"一条消息。
    # Annotated[list, add_messages] 就是在告诉 LangGraph：这个字段要追加，不要覆盖。
    messages: Annotated[list, add_messages]
    spots: str           # 景点推荐结果（自然语言文本，含每个景点的亮点与适配理由）

    # ===== 规划 Agent 产出 =====
    itinerary: str       # 按天排好的行程安排（自然语言文本，供后续 Agent 阅读）

    # ===== 路线 Agent 产出 =====
    route_plan: str      # 带交通衔接和时刻的详细路线（自然语言文本，全流程的最终产出）
    # 字段名用 route_plan 而不是 route，是为了不和 route.py（路由函数文件）撞名
