"""条件边的路由函数：图"往哪走"的判断集中在这里。

普通边（add_edge）一定走；条件边（add_conditional_edges）看情况走——
具体看什么，由本文件的函数决定：读 state → 返回下一个节点的名字。
"""

from langgraph.graph import END

from agent_state import AgentState


MAX_MESSAGES = 20


def should_continue(state: AgentState) -> str:
    """景点 Agent 的条件边：这一轮还想不想调工具？

    返回 "tools"   —— 还想搜，去工具节点执行后回到景点 Agent
    返回 "planner" —— 不搜了（清单已写好），交给规划节点排行程
    返回 END       —— 兜底收工（循环失控时）
    """
    messages = state["messages"]

    # 保险：循环失控（如工具反复报错）时强行收工，避免程序卡死
    if len(messages) >= MAX_MESSAGES:
        return END

    if getattr(messages[-1], "tool_calls", None):
        return "tools"

    return "planner"
