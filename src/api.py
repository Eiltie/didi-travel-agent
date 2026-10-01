"""网页后端的入口：把 Agent 的能力，变成浏览器能请求的"接口"。

【这个文件现在走到第 4 步了：结果一段一段往外冒（流式）】
    第 1 步：GET  /api/ping  —— 证明服务活着
    第 2 步：POST /api/chat  —— 收一句话 → 跑图 → 一次性把结果发回去
    第 3 步：GET  /          —— 把 static/page.html 发给浏览器
    第 4 步：/api/chat 改成流式 —— 图每跑完一个节点，就推一段回去（就是这一步）

"流式"和以前唯一的区别是【什么时候发】：

    以前：图全跑完（约 1 分钟）→ 攒成一个大结果 → 一次性 return
    现在：跑完天气就发天气、跑完景点就发景点 → 边跑边 yield

图本身（4 个节点、其余 8 个文件）一个字都没动——变的只是这个文件"交货的方式"。

【再往后一步（最新）：从一个字一个字冒】
    第 4 步的流式，粒度是"一个节点跑完推一整段"。段内部还是黑的：
    跑完景点到跑完行程之间隔了 70 多秒，屏幕上一个新字都没有——因为
    planner 要把整段想完才交货，还是"干等"。

    这次换成 LangGraph 自带的 messages 模式：图里每个 LLM 每吐一个字，
    就立刻推一个字出去。图还是一动不动，变的依旧是"交货的方式"。

    但 LLM 吐的东西不全都能给用户看（有解析用的 JSON、有"搜什么"的工具参数），
    所以配了三道过滤 + 一个标签约定——具体在 event_stream 的注释里。
    标签需要 agent_node.py 配合一毫米：天气节点里"解析 JSON"那次调用
    贴一个 internal 标签（就这一处），别的图内文件一个字没动。

怎么启动它（两种都行，在项目根目录下执行）：

    ① 直接跑（简单，但改了代码要手动重启）
        venv\\Scripts\\python.exe src\\api.py

    ② 用 uvicorn 跑（多打几个字，但改了代码自动重启）
        venv\\Scripts\\python.exe -m uvicorn api:app --reload --app-dir src

两种方式跑起来后，浏览器打开：
    http://127.0.0.1:8000/docs      ← 在这里点着测接口
    http://127.0.0.1:8000/api/ping  ← 第 1 步那个，还在

想看"流式是不是真的流"，用下面这条命令（在 Git Bash 里、项目根目录下）。
之所以不用 curl：Windows 的 Git Bash 里 curl 传中文会报 "error parsing the body"，
换成 Python 发这同一个请求就没这个毛病（这也是之前踩过的坑）。

    PYTHONUTF8=1 venv/Scripts/python.exe - <<'PY'
    import json, urllib.request
    body = json.dumps({"query": "下周想去杭州玩3天，带老人", "history": []}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/api/chat", body,
                                 {"Content-Type": "application/json"})
    for line in urllib.request.urlopen(req):
        print(line.decode("utf-8"), end="", flush=True)
    PY

看到 data: {...} 一条条冒出来就是成了。现在大量 TOKEN 事件之间几乎不停顿
（一个字一条），"停一下"的长停顿只出现在等工具结果的时候——那正是应该的。

注意（最新）：这次前端也要跟着升级——TOKEN / DONE / STEP 这几个新事件，
page.html 现在还不认识，所以这个文件改完【先别用网页测】，用下面那条
Python 命令直接看事件流；等 page.html 也改完，网页上的打字机效果才完整。

不管用哪种方式启动，底下那个 if __name__ == "__main__" 那段
决定了能不能"直接跑"——这就是 main.py 能直接跑、别的文件不能的原因。
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

from create_graph import build_graph
from main import understand_query


# ===== 应用本体 =====

app = FastAPI(title="嘀嘀智能出行")
travel_graph = build_graph()


class ChatRequest(BaseModel):
    query: str
    history: list[str] = []


# ===== 流式输出（SSE）=====

def sse(event: dict) -> str:
    """把一段数据打包成一条 SSE 消息（结尾那个空行就是"这条完了"的句号）。"""
    # ensure_ascii=False：中文原样发，不转成 \uXXXX（不然命令行里看到一片乱码）
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


SECTION_KEYS = ("weather", "spots", "itinerary", "route_plan")

# 节点名 → 前端那一段的 key。逐字流出来的文字碎片，凭这张表知道该贴到页面上哪一段。
# 对不上表的节点（比如 tools）直接跳过——它们的输出不是给用户看的正文。
NODE_TO_KEY = {
    "weather": "weather",
    "spots_agent": "spots",      # 注意：节点名和段名不一样，别搞混
    "planner": "itinerary",      # "安排行程"节点的产物对应前端"行程安排"段
    "route_agent": "route_plan",
}

# "表面没动静"的时段（模型在思考、工具在干活）用一行小字把空档填住。
# 前端收到 STEP 就在对应段落里显示这行字，该段的正文一冒头它就下架。
STEP_TEXT = {
    "weather": "正在查询天气数据…",
    "spots": "正在搜集景点资料…",
    "itinerary": "正在安排行程…",
    "route_plan": "正在查询路线…",
}

# 图的执行顺序是固定的（天气 → 景点 → 行程 → 路线），所以"这段跑完轮到谁"
# 可以直接查表，不用等下段自己的信号。以后图中途加了节点，改这里。
NEXT_KEY = {
    "weather": "spots",
    "spots": "itinerary",
    "itinerary": "route_plan",
    # route_plan 是最后一段，后面没有了
}


def step(key: str) -> str:
    """打包一条"正在忙…"的小字事件。"""
    return sse({"kind": "STEP", "key": key, "text": STEP_TEXT[key]})


def event_stream(query: str, history: list[str]):
    """生成器：跑一次图，LLM 每吐一点字就往外送一条 SSE 消息。

    "生成器"是什么？函数里只要出现了 yield，它的脾气就变了：
        普通函数：从头跑到尾，return 一个值，然后函数就没了
        生成器：  跑到 yield 就【暂停】，把值交出去；外面再来要，才从原地继续往下跑
    所以它不是"一次算完"，而是"外面要一次、它才算一步"。
    流式的原理就在这儿：前端一边收，图一边往下跑。
    """
    kind, content = understand_query(history, query)

    if kind == "CHAT":
        yield sse({"kind": "CHAT", "message": content})
        return          # 生成器里的 return 只有一个意思："我说完了"，流到此结束

    yield sse({"kind": "START"})
    yield step("weather")       # 开工的第一个节点必是天气，先把小字亮出来
    waiting = {"weather"}       # 小字正亮着的段；该段的正文一冒头，就把它下架

    try:
        # stream_mode 这次传两个，每次吐出来的东西是 (是哪种, 内容) 一对：
        #   "updates"  = 老规矩：某个节点整个跑完，推一次它的完整产出
        #   "messages" = 新加的：图里每个 LLM 每吐一小段字，就推一小段
        # 图本身（包括那四个节点）一个字没改，多出来的能力全来自这一个参数。
        for mode, payload in travel_graph.stream({
            "user_query": content,
            "destination": "",
            "days": 0,
            "start_date": "",
            "weather": "",
            "messages": [],      # 景点 Agent 的 ReAct"草稿纸"，开场是空的
            "spots": "",
            "itinerary": "",
            "route_plan": "",
        }, stream_mode=["updates", "messages"]):

            # ① 逐字的那种：payload 是 (一小段字, 这条字的情报) 一对
            if mode == "messages":
                chunk, meta = payload
                key = NODE_TO_KEY.get(meta.get("langgraph_node"))

                # 下面四关全过了才发给前端，少一关都不行：
                #   关1 是本图认识的节点——tools 这类对不上表，跳过
                #   关2 只有 LLM 吐的"碎片"才算正文——messages 模式还会冒出
                #       SystemMessage 这种【整条】消息（比如景点节点的提示词全文），
                #       那是程序内部的东西，不能见光
                #   关3 工具参数不算正文——ReAct 中间轮里 LLM 在决定"搜什么"，
                #       吐的是 {"query": ...} 这种参数，一律不放行
                #   关4 贴了 internal 标签的也不算——天气节点里"解析 JSON"那次
                #       调用贴了它（见 agent_node.py）。实测过：不挡的话
                #       "{"destination": "杭州"..." 会原样流到用户眼皮底下
                if not isinstance(chunk, AIMessageChunk) or not key:
                    continue
                if chunk.tool_call_chunks:
                    continue
                if not isinstance(chunk.content, str) or not chunk.content:
                    continue
                if "internal" in (meta.get("tags") or []):
                    continue

                waiting.discard(key)            # 正文来了，小字下班
                yield sse({"kind": "TOKEN", "key": key, "text": chunk.content})

            # ② 整段的那种：某个节点跑完了，推它的完整产出（事件名从 TRAVEL 改成 DONE）
            #    前端收到 DONE 会拿这份"权威全文"重渲染定稿——万一逐字拼出来的
            #    版本有偏差，这里纠正回来。
            else:
                for _, update in payload.items():
                    for key in SECTION_KEYS:
                        if key in update:
                            waiting.discard(key)
                            yield sse({"kind": "DONE", "key": key, "text": update[key]})
                            nxt = NEXT_KEY.get(key)
                            if nxt:             # 这段跑完，就轮到下一段忙了——小字顶上
                                waiting.add(nxt)
                                yield step(nxt)
    except Exception as e:
        yield sse({"kind": "ERROR", "message": f"这次没跑通：{e}"})


# ===== 接口 =====

@app.get("/api/ping")
def ping():
    """最简单的接口：浏览器一访问，就回一句话。"""
    return {"ok": True, "message": "服务活着"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """收一句话，返回一条 SSE 流（流里的内容由上面的 event_stream 一段段生成）。"""
    return StreamingResponse(
        event_stream(req.query, req.history),
        media_type="text/event-stream",   # 告诉浏览器"这是一条事件流"，不是一包 JSON
        headers={
            "Cache-Control": "no-cache",  # 这条流别缓存，每次都要新的
            "X-Accel-Buffering": "no",    # 以后要是前面挂了 nginx，让它别攒着、直接转发
        },
    )


# ===== 静态文件与首页 =====

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

NO_CACHE = {"Cache-Control": "no-cache"}


class NoCacheStatic(StaticFiles):
    """在原版 StaticFiles 外面套一层，只为了给发出去的文件补上那个响应头。

    参数写成 *args / **kwargs，而不是照着源码把签名抄一遍 ——
    这样就不依赖 Starlette 的内部写法了，它换版本改了参数这里也不会坏。
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response


app.mount("/static", NoCacheStatic(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    """首页：把 page.html 原样发给浏览器。"""
    return FileResponse(STATIC_DIR / "page.html", headers=NO_CACHE)


# ===== 直接运行 =====

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
