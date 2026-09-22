"""构建图：把各个 Agent 节点组装成一张 LangGraph 流程图。"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent_node import planner_node, route_agent_node, spots_agent_node, weather_node
from agent_state import AgentState
from agent_tools import web_search
from route import should_continue


tools_node = ToolNode([web_search], handle_tool_errors=True)


def build_graph():
    """组装并编译出行助手流程图。"""
    graph = StateGraph(AgentState)            # ① 声明用哪个 state

    graph.add_node("weather", weather_node)   # ② 挂上五个节点
    graph.add_node("spots_agent", spots_agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("planner", planner_node)
    graph.add_node("route_agent", route_agent_node)

    graph.add_edge(START, "weather")          # ③ 连边
    graph.add_edge("weather", "spots_agent")  #    天气 → 景点大脑
    graph.add_conditional_edges("spots_agent", should_continue)   # 岔路口：还搜不搜 / 去排期？
    graph.add_edge("tools", "spots_agent")    #    搜完回到大脑，形成循环
    graph.add_edge("planner", "route_agent")  #    排完行程 → 细化路线
    graph.add_edge("route_agent", END)        #    路线出完 → 收工

    return graph.compile()                    # ④ 编译

