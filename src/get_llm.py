"""大模型的获取：全项目统一从这里拿 DeepSeek 实例。"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 显式指定 .env 路径：本文件在 src/ 下，上一级就是项目根目录
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)


@lru_cache(maxsize=None)
def get_llm() -> ChatOpenAI:
    """返回配好 DeepSeek 的模型实例（全局复用同一个）。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(f"没读到 DEEPSEEK_API_KEY，请检查 {_ENV_PATH}")

    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )
