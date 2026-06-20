"""Web search via Tavily — wrapped with graceful degradation."""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def _get_client() -> TavilyClient:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set in .env")
    return TavilyClient(api_key=key)


def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web and return normalized results.

    Returns:
        List of dicts with keys: title, url, content, score
        Returns empty list on failure (graceful degradation).
    """
    try:
        client = _get_client()
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # 'advanced' is slower and pricier
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:800],  # truncate to control tokens
                "score": r.get("score", 0.0),
            }
            for r in resp.get("results", [])
        ]
    except Exception as e:
        # Graceful degradation: log but don't crash the graph
        print(f"[search] failed: {type(e).__name__}: {e}")
        return []