"""Query Agent: searches the web based on host's directive, then summarizes."""
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage

from tools.llm import get_llm
from tools.search import search_web
from graph.state import OverallState


# ──────────────────────────────────────────────────────────────────────────
# Subgraph-internal state. Lives only inside the Query subgraph.
# ──────────────────────────────────────────────────────────────────────────
class QueryState(TypedDict):
    topic: str
    directive: str
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    summary: str
    sources: List[str]


# ──────────────────────────────────────────────────────────────────────────
# Node 1: turn topic + directive into 2-3 concrete search queries
# ──────────────────────────────────────────────────────────────────────────
QUERY_PLANNING_PROMPT = """You are a search query planner.

Topic: {topic}
Host directive: {directive}

Generate 2-3 concrete, specific Chinese search queries that will help address
the directive. Prefer recent angles (news, performance reports, comparisons).

Output STRICT JSON list of strings, nothing else:
["query 1", "query 2", "query 3"]
"""


def plan_queries(state: QueryState) -> dict:
    llm = get_llm(tier="default", temperature=0.3)
    prompt = QUERY_PLANNING_PROMPT.format(
        topic=state["topic"],
        directive=state.get("directive") or "Provide a broad overview.",
    )
    resp = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    # Strip markdown fences
    if resp.startswith("```"):
        parts = resp.split("```")
        if len(parts) >= 2:
            resp = parts[1].lstrip("json").strip()

    queries: List[str] = []
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, list):
            queries = [str(q) for q in parsed if isinstance(q, str)][:3]
    except json.JSONDecodeError:
        pass

    # Fallback: just use the topic itself
    if not queries:
        queries = [state["topic"]]

    return {"search_queries": queries}


# ──────────────────────────────────────────────────────────────────────────
# Node 2: execute searches in parallel-ish (sequential for now, simpler)
# ──────────────────────────────────────────────────────────────────────────
def execute_searches(state: QueryState) -> dict:
    all_results = []
    for q in state["search_queries"]:
        results = search_web(q, max_results=3)
        all_results.extend(results)

    # Dedup by URL
    seen = set()
    deduped = []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)

    return {"search_results": deduped}


# ──────────────────────────────────────────────────────────────────────────
# Node 3: summarize search results into a structured paragraph
# ──────────────────────────────────────────────────────────────────────────
SUMMARY_PROMPT = """You are a research analyst.

Topic: {topic}
Host directive: {directive}

Search results:
{results}

Write a focused 200-300 word summary in Chinese that:
1. Directly addresses the host's directive
2. Cites concrete facts from the search results (don't invent)
3. Notes if evidence is thin or conflicting

If search results are empty or irrelevant, say so honestly in 1-2 sentences.
"""


def summarize(state: QueryState) -> dict:
    results = state.get("search_results", [])

    if not results:
        return {
            "summary": "本轮搜索未返回有效结果。可能原因：检索词过窄、目标信息时效性问题或 API 限制。",
            "sources": [],
        }

    results_text = "\n\n".join(
        f"[{i+1}] {r['title']}\n{r['content']}\n来源: {r['url']}"
        for i, r in enumerate(results[:8])  # cap at 8 to control tokens
    )

    llm = get_llm(tier="default", temperature=0.4)
    prompt = SUMMARY_PROMPT.format(
        topic=state["topic"],
        directive=state.get("directive") or "Provide a broad overview.",
        results=results_text,
    )
    summary = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    return {
        "summary": summary,
        "sources": [r["url"] for r in results[:8]],
    }


# ──────────────────────────────────────────────────────────────────────────
# Build the subgraph
# ──────────────────────────────────────────────────────────────────────────
def build_query_subgraph():
    g = StateGraph(QueryState)
    g.add_node("plan_queries", plan_queries)
    g.add_node("execute_searches", execute_searches)
    g.add_node("summarize", summarize)

    g.add_edge(START, "plan_queries")
    g.add_edge("plan_queries", "execute_searches")
    g.add_edge("execute_searches", "summarize")
    g.add_edge("summarize", END)

    return g.compile()


# ──────────────────────────────────────────────────────────────────────────
# Wrapper: adapts subgraph to the OverallState contract
# ──────────────────────────────────────────────────────────────────────────
_subgraph = None


def query_agent_node(state: OverallState) -> dict:
    """Entry point called by the top-level graph."""
    global _subgraph
    if _subgraph is None:
        _subgraph = build_query_subgraph()

    sub_initial: QueryState = {
        "topic": state["topic"],
        "directive": state.get("host_directive") or "",
        "search_queries": [],
        "search_results": [],
        "summary": "",
        "sources": [],
    }

    result = _subgraph.invoke(sub_initial)

    return {
        "agent_results": [{
            "agent_name": "query",
            "content": result["summary"],
            "sources": result["sources"],
            "confidence": 0.7 if result["sources"] else 0.3,
        }],
        "total_tokens": 600,  # rough estimate (planning + summary)
    }
