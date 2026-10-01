"""网页后端入口：把 Agent 能力包装成浏览器可请求的接口。

启动（项目根目录，二选一，端口 8001）：
    venv\\Scripts\\python.exe src\\api.py  # 直接跑
    venv\\Scripts\\python.exe -m uvicorn api:app --reload --app-dir src --port 8001  # 改代码自动重启

打开：
    http://127.0.0.1:8001/docs      接口测试页
    http://127.0.0.1:8001/api/ping  存活检测

/api/chat 为 SSE 流式接口，事件有三类：TOKEN（LLM 逐字吐字）/ STEP（节点进度）/ DONE（整段定稿）。

想看原始事件流，用下面这条命令（Git Bash 里 curl 传中文会报错，改用 Python 发请求）：

    PYTHONUTF8=1 venv/Scripts/python.exe - <<'PY'
    import json, urllib.request
    body = json.dumps({"query": "下周想去杭州玩3天，带老人", "history": []}).encode()
    req = urllib.request.Request("http://127.0.0.1:8001/api/chat", body,
                                 {"Content-Type": "application/json"})
    for line in urllib.request.urlopen(req):
        print(line.decode("utf-8"), end="", flush=True)
    PY
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
    """把事件数据打包成一条 SSE 消息（末尾空行即本条结束）。"""
    # ensure_ascii=False：中文原样发送，不转成 \uXXXX（否则命令行下一片乱码）
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


SECTION_KEYS = ("weather", "spots", "itinerary", "route_plan")

# 节点名 → 前端段落的 key：逐字碎片凭这张表决定贴到页面哪一段。
# 表外节点（如 tools）跳过——其输出不是给用户看的正文。
NODE_TO_KEY = {
    "weather": "weather",
    "spots_agent": "spots",      # 节点名与段名不一致，勿混
    "planner": "itinerary",      # "安排行程"节点对应前端"行程安排"段
    "route_agent": "route_plan",
}

# 静默时段（模型思考、工具执行）的进度提示文字。
# 前端收到 STEP 事件就显示在对应段落里，该段正文一到即清除。
STEP_TEXT = {
    "weather": "正在查询天气数据…",
    "spots": "正在搜集景点资料…",
    "itinerary": "正在安排行程…",
    "route_plan": "正在查询路线…",
}

# 图按固定顺序执行（天气 → 景点 → 行程 → 路线），"这段跑完轮到谁"直接查表即可。
# 以后图中加节点，改这里。
NEXT_KEY = {
    "weather": "spots",
    "spots": "itinerary",
    "itinerary": "route_plan",
    # route_plan 是最后一段，无后继
}


def step(key: str) -> str:
    """打包一条进度提示（STEP）事件。"""
    return sse({"kind": "STEP", "key": key, "text": STEP_TEXT[key]})


def event_stream(query: str, history: list[str]):
    """生成器：跑一次图，每吐一点字就往外送一条 SSE 消息——前端边收、图边跑。"""
    kind, content = understand_query(history, query)

    if kind == "CHAT":
        yield sse({"kind": "CHAT", "message": content})
        return          # 结束这条流

    yield sse({"kind": "START"})
    yield step("weather")       # 开工的第一个节点必是天气，先把小字亮出来
    waiting = {"weather"}       # 小字正亮着的段；该段的正文一冒头，就把它下架

    try:
        # stream_mode 传两个，每次吐出 (类型, 内容) 一对：
        #   "updates"  = 节点整体跑完，推一次完整产出
        #   "messages" = 每个 LLM 吐一小段字，就推一小段
        # 图本身不用改，逐字流的能力全来自这个参数。
        for mode, payload in travel_graph.stream({
            "user_query": content,
            "destination": "",
            "days": 0,
            "start_date": "",
            "weather": "",
            "messages": [],      # 景点 Agent 的 ReAct 中间消息，初始为空
            "spots": "",
            "itinerary": "",
            "route_plan": "",
        }, stream_mode=["updates", "messages"]):

            # ① 逐字流入：payload = (文字片段, 元信息)
            if mode == "messages":
                chunk, meta = payload
                key = NODE_TO_KEY.get(meta.get("langgraph_node"))

                # 四道过滤全部通过才发给前端：
                #   1) 节点在 NODE_TO_KEY 表内 —— tools 等表外节点跳过
                #   2) 是 AIMessageChunk —— messages 模式也会带出 SystemMessage 整条消息
                #   3) 无工具调用参数 —— ReAct 中间轮吐的是 {"query": ...}
                #   4) 无 internal 标签 —— 天气节点解析 JSON 的调用贴了它（见 agent_node.py），
                #      缺这道过滤时 {"destination": "杭州"...} 会直接流给用户
                if not isinstance(chunk, AIMessageChunk) or not key:
                    continue
                if chunk.tool_call_chunks:
                    continue
                if not isinstance(chunk.content, str) or not chunk.content:
                    continue
                if "internal" in (meta.get("tags") or []):
                    continue

                waiting.discard(key)            # 正文到达，清除该段提示
                yield sse({"kind": "TOKEN", "key": key, "text": chunk.content})

            # ② 整段流入：节点跑完，推完整产出（事件名 TRAVEL 已改为 DONE）。
            #    前端收到 DONE 用它重渲染定稿，修正逐字拼接可能产生的偏差。
            else:
                for _, update in payload.items():
                    for key in SECTION_KEYS:
                        if key in update:
                            waiting.discard(key)
                            yield sse({"kind": "DONE", "key": key, "text": update[key]})
                            nxt = NEXT_KEY.get(key)
                            if nxt:             # 该段完成 → 亮下一段提示
                                waiting.add(nxt)
                                yield step(nxt)
    except Exception as e:
        yield sse({"kind": "ERROR", "message": f"这次没跑通：{e}"})


# ===== 接口 =====

@app.get("/api/ping")
def ping():
    """存活检测：访问即回一句话。"""
    return {"ok": True, "message": "服务活着"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """收一句话，返回 event_stream 生成的 SSE 流。"""
    return StreamingResponse(
        event_stream(req.query, req.history),
        media_type="text/event-stream",   # 声明为事件流（而非一次性 JSON 响应）
        headers={
            "Cache-Control": "no-cache",  # 流不缓存
            "X-Accel-Buffering": "no",    # 若前面挂 nginx：禁止攒包，逐条转发
        },
    )


# ===== 静态文件与首页 =====

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

NO_CACHE = {"Cache-Control": "no-cache"}


class NoCacheStatic(StaticFiles):
    """包一层 StaticFiles，只为给响应补上 no-cache 头。

    参数用 *args / **kwargs 透传，不抄 Starlette 的签名——它换版本改了参数，这里也不会坏。
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
