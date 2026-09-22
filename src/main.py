"""图入口：运行整个出行助手（支持多轮对话 + 打招呼聊天）。

多轮是怎么做到的：图本身完全不知道有"历史"这回事。
是这里先用一次 LLM 把「之前聊过的 + 这一句」看懂——
是出行需求，就合并成一句完整需求喂给图；是闲聊，就直接回一句，不跑图。
所以除了这个文件，项目其他部分一个字没改。
"""

from create_graph import build_graph
from get_llm import get_llm
from total_prompts import UNDERSTAND_QUERY_PROMPT

DEFAULT_QUERY = "下周想去杭州玩3天，带老人"
EXIT_WORDS = {"exit", "quit", "q", "退出"}

# 图里的节点名是英文，这里给它们起个给人看的名字。
# 注意 spots_agent 会出现好几次——那是它在循环里反复搜，不是出错。
NODE_LABELS = {
    "weather": "查询天气",
    "spots_agent": "推荐景点",
    "tools": "搜索景点信息",
    "planner": "安排行程",
    "route_agent": "规划路线",
}


def understand_query(history: list[str], query: str) -> tuple[str, str]:
    """看懂这一句：是出行需求（TRAVEL）还是闲聊（CHAT）。

    返回 (类型, 内容)：TRAVEL 时内容是一句完整的出行需求，CHAT 时内容是给用户的回复。
    """
    prompt = (UNDERSTAND_QUERY_PROMPT
              .replace("{history}", "\n".join(history) or "（还没聊过）")
              .replace("{query}", query))
    reply = get_llm().invoke(prompt).content.strip()

    # 约定：第一行只写 TRAVEL 或 CHAT，第二行开始是正文
    lines = reply.split("\n", 1)
    kind = lines[0].strip().upper()
    body = lines[1].strip() if len(lines) > 1 else ""

    if kind.startswith("CHAT"):
        return "CHAT", body
    return "TRAVEL", body or reply      # 模型没守格式？当出行需求处理，别卡住


def show(result: dict) -> None:
    """把这一轮的结果打印出来。"""
    print("\n" + "=" * 56)
    print(f"目的地：{result['destination']}    天数：{result['days']}    出发：{result['start_date']}")
    print("=" * 56)
    print("\n【天气】\n" + result["weather"])
    print("\n【景点推荐】\n" + result["spots"])
    print("\n【行程安排】\n" + result["itinerary"])
    print("\n【详细路线】\n" + result["route_plan"])


def main():
    app = build_graph()
    history = []          # 用户说过的话，一句一条

    print("嘀嘀智能出行 —— 可以接着上一句继续聊（输入 exit 退出）")

    while True:
        query = input("\n你：").strip()

        if query.lower() in EXIT_WORDS:
            print("再见！")
            break
        if not query:
            if history:
                continue                    # 已经聊过了，空回车就跳过
            query = DEFAULT_QUERY           # 第一次空回车 → 用默认需求，方便试

        # 看懂这一句：是出行需求，还是打招呼闲聊
        kind, content = understand_query(history, query)

        if kind == "CHAT":
            print(f"\n助手：{content}")
            history.append(query)
            continue                        # 闲聊不跑图，直接等下一句

        print(f"（理解为：{content}）")

        print("\n正在规划中…")
        try:
            # 用 stream 而不是 invoke：invoke 要等整张图跑完才返回，中间一片黑；
            # stream 是每跑完一个节点就吐一次，于是能一行一行报告进度。
            result = {}
            for chunk in app.stream({
                "user_query": content,
                "destination": "",
                "days": 0,
                "start_date": "",
                "weather": "",
                "messages": [],      # ReAct 循环的"草稿纸"，开场是空的，由景点节点自己填
                "spots": "",         # 景点 Agent 的产出，开场是空的
                "itinerary": "",     # 规划 Agent 的产出，开场是空的
                "route_plan": "",    # 路线 Agent 的产出，开场是空的
            }):
                for node_name, update in chunk.items():
                    print(f"  √ {NODE_LABELS.get(node_name, node_name)}")
                    # 把每个节点的产出攒起来，攒完就是完整结果。
                    # （注意 messages 在这里会被覆盖成最后一条，展示用不上它，真要它得另想办法）
                    result.update(update)

            show(result)
        except Exception as e:
            # 兜底：万一哪一步出了意外，也别让程序退出，下一句还能接着聊
            print(f"\n（这次没跑通：{e}）")
            print("换个说法再试试？比如「下周想去杭州玩3天」")

        history.append(query)


if __name__ == "__main__":
    main()
