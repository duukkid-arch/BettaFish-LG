"""LLM factory with tiered model selection."""
import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi

load_dotenv()


@lru_cache(maxsize=4)
def get_llm(tier: str = "default", temperature: float = 0.3):
    """
    Get a cached LLM instance.

    tier='default'  -> qwen-turbo  (cheap, for routing and intermediate agents)
    tier='premium'  -> qwen-plus   (better, for final synthesis)
    """
    if tier == "premium":
        model = os.getenv("REPORT_MODEL", "qwen-plus")
    else:
        model = os.getenv("DEFAULT_MODEL", "qwen-turbo")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set in .env")

    return ChatTongyi(
        model=model,
        temperature=temperature,
        dashscope_api_key=api_key,
    )
