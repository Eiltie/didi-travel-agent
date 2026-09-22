"""条件边的路由函数：图"往哪走"的判断集中在这里。

普通边（add_edge）表示"一定走这条路"；条件边（add_conditional_edges）表示"看情况走哪条"。
而"看情况"具体看什么、怎么判断，就由本文件的函数说了算：
    读一眼 state → 返回一个字符串 → LangGraph 拿它当"下一个节点的名字"

条件边的出口可以不止两个。下面这个函数就有三个去向：
    "tools"   → 去工具节点搜一轮
    "planner" → 不搜了，交给规划节点排行程
    END       → 兜底收工（只有循环失控时才会走到）
"""

from langgraph.graph import END

from agent_state import AgentState


MAX_MESSAGES = 20


def should_continue(state: AgentState) -> str:
    """景点 Agent 的条件边：这一轮它还想不想调工具？

    返回 "tools"   —— 还想搜，去工具节点执行，执行完再回到景点 Agent
    返回 "planner" —— 不想搜了，说明清单已写好，交给规划节点排行程
    返回 END       —— 只在循环失控时兜底收工
    """
    messages = state["messages"]

    # 保险：万一循环失控（比如工具反复报错），强行收工，别让程序卡死
    if len(messages) >= MAX_MESSAGES:
        return END


    if getattr(messages[-1], "tool_calls", None):
        return "tools"

    return "planner"
