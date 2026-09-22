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

看到 data: {...} 一条条冒出来、每条之间还停一下（那是图在跑下一个节点），就是成了。
要是一口气全吐出来，说明哪里还在攒——那就没流式成功。

注意：这一步只改了后端。static/page.html 还停在老写法（等一整包 JSON），
所以这会儿点网页会打不开结果——下一步就修好它。

不管用哪种方式启动，底下那个 if __name__ == "__main__" 那段
决定了能不能"直接跑"——这就是 main.py 能直接跑、别的文件不能的原因。
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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


def event_stream(query: str, history: list[str]):
    """生成器：跑一次图，跑出一个节点就吐一条 SSE 消息出去。

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

    try:
        for chunk in travel_graph.stream({
            "user_query": content,
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
                for key in SECTION_KEYS:
                    if key in update:
                        yield sse({"kind": "TRAVEL", "key": key, "text": update[key]})
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

    uvicorn.run(app, host="127.0.0.1", port=8000)
