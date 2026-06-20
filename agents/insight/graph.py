"""Insight Agent: consumes prior agents' outputs and produces deep multi-dimensional analysis."""
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage

from tools.llm import get_llm
from graph.state import OverallState


# ──────────────────────────────────────────────────────────────────────────
# Subgraph-internal state
# ──────────────────────────────────────────────────────────────────────────
class InsightState(TypedDict):
    topic: str
    directive: str
    prior_evidence: str         # concatenated content from previous agents
    dimensions: List[str]       # e.g. ["技术架构", "市场反响", "竞争格局"]
    sentiment_map: Dict[str, Dict[str, Any]]  # dim -> {sentiment, confidence, rationale}
    insight: str


# ──────────────────────────────────────────────────────────────────────────
# Node 1: extract 3-5 analytical dimensions from prior evidence
# ──────────────────────────────────────────────────────────────────────────
DIMENSION_PROMPT = """You are an analytical framework designer.

Topic: {topic}
Host directive: {directive}

Prior evidence (from earlier agents):
{evidence}

Extract 3-5 analytical dimensions that are worth deep analysis. Dimensions should be:
- Specific to this topic (not generic like "advantages" or "disadvantages")
- Grounded in the evidence (don't invent dimensions with no support)
- Diverse (don't list 3 dimensions that are basically the same thing)

Output STRICT JSON list of Chinese strings, nothing else:
["维度1", "维度2", "维度3"]
"""


def extract_dimensions(state: InsightState) -> dict:
    llm = get_llm(tier="default", temperature=0.3)
    prompt = DIMENSION_PROMPT.format(
        topic=state["topic"],
        directive=state.get("directive") or "Provide deep analysis.",
        evidence=state.get("prior_evidence") or "(no prior evidence)",
    )
    resp = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    if resp.startswith("```"):
        parts = resp.split("```")
        if len(parts) >= 2:
            resp = parts[1].lstrip("json").strip()

    dimensions: List[str] = []
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, list):
            dimensions = [str(d) for d in parsed if isinstance(d, str)][:5]
    except json.JSONDecodeError:
        pass

    # Fallback if extraction fails
    if not dimensions:
        dimensions = ["核心特点", "市场反响", "潜在风险"]

    return {"dimensions": dimensions}


# ──────────────────────────────────────────────────────────────────────────
# Node 2: aspect-level sentiment analysis (one LLM call, batched)
# ──────────────────────────────────────────────────────────────────────────
SENTIMENT_PROMPT = """You are an aspect-based sentiment analyzer.

Topic: {topic}
Evidence:
{evidence}

For each dimension below, judge the sentiment expressed in the evidence:
{dimensions}

For each dimension, output:
- sentiment: one of "positive", "negative", "neutral", "mixed"
- confidence: float 0-1 (how clear the evidence is on this dimension)
- rationale: 1 sentence in Chinese explaining the judgment

Output STRICT JSON, nothing else:
{{
  "维度1": {{"sentiment": "positive", "confidence": 0.8, "rationale": "..."}},
  "维度2": {{"sentiment": "mixed", "confidence": 0.6, "rationale": "..."}}
}}
"""


def analyze_sentiment(state: InsightState) -> dict:
    dims_text = "\n".join(f"- {d}" for d in state["dimensions"])

    llm = get_llm(tier="default", temperature=0.2)
    prompt = SENTIMENT_PROMPT.format(
        topic=state["topic"],
        evidence=state.get("prior_evidence") or "(no prior evidence)",
        dimensions=dims_text,
    )
    resp = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    if resp.startswith("```"):
        parts = resp.split("```")
        if len(parts) >= 2:
            resp = parts[1].lstrip("json").strip()

    sentiment_map: Dict[str, Dict[str, Any]] = {}
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, dict):
            for dim, data in parsed.items():
                if isinstance(data, dict):
                    sentiment_map[str(dim)] = {
                        "sentiment": str(data.get("sentiment", "neutral")),
                        "confidence": float(data.get("confidence", 0.5)),
                        "rationale": str(data.get("rationale", "")),
                    }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: blank entries for each dimension
    if not sentiment_map:
        sentiment_map = {
            d: {"sentiment": "neutral", "confidence": 0.3, "rationale": "证据不足以判断"}
            for d in state["dimensions"]
        }

    return {"sentiment_map": sentiment_map}


# ──────────────────────────────────────────────────────────────────────────
# Node 3: synthesize into a coherent insight paragraph
# ──────────────────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = """You are a senior sentiment analyst.

Topic: {topic}
Host directive: {directive}

You have analyzed these dimensions:
{sentiment_summary}

Write a 250-350 word深度洞察 in Chinese that:
1. Goes BEYOND restating the evidence — interpret WHY each sentiment is what it is
2. Connects dimensions (e.g. "positive on X is fueling concern on Y")
3. Flags any contradictions or nuances the prior agents may have missed
4. Avoids generic phrases like "总的来说"; be concrete

Do NOT cite URLs (your job is interpretation, not citation).
"""


def synthesize_insight(state: InsightState) -> dict:
    sentiment_lines = []
    for dim, data in state["sentiment_map"].items():
        sentiment_lines.append(
            f"- {dim}: 情感={data['sentiment']}, 置信度={data['confidence']:.2f}, "
            f"依据={data['rationale']}"
        )
    sentiment_summary = "\n".join(sentiment_lines)

    llm = get_llm(tier="default", temperature=0.5)
    prompt = SYNTHESIS_PROMPT.format(
        topic=state["topic"],
        directive=state.get("directive") or "Provide deep analysis.",
        sentiment_summary=sentiment_summary,
    )
    insight = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    return {"insight": insight}


# ──────────────────────────────────────────────────────────────────────────
# Build the subgraph
# ──────────────────────────────────────────────────────────────────────────
def build_insight_subgraph():
    g = StateGraph(InsightState)
    g.add_node("extract_dimensions", extract_dimensions)
    g.add_node("analyze_sentiment", analyze_sentiment)
    g.add_node("synthesize_insight", synthesize_insight)

    g.add_edge(START, "extract_dimensions")
    g.add_edge("extract_dimensions", "analyze_sentiment")
    g.add_edge("analyze_sentiment", "synthesize_insight")
    g.add_edge("synthesize_insight", END)

    return g.compile()


# ──────────────────────────────────────────────────────────────────────────
# Wrapper: adapts subgraph to OverallState
# ──────────────────────────────────────────────────────────────────────────
_subgraph = None


def insight_agent_node(state: OverallState) -> dict:
    """Entry point called by the top-level graph."""
    global _subgraph
    if _subgraph is None:
        _subgraph = build_insight_subgraph()

    # Collect prior evidence from all earlier agent_results
    prior_evidence_pieces = []
    for r in state.get("agent_results", []):
        prior_evidence_pieces.append(f"[{r['agent_name']}]\n{r['content']}")
    prior_evidence = "\n\n".join(prior_evidence_pieces) or "(none)"

    sub_initial: InsightState = {
        "topic": state["topic"],
        "directive": state.get("host_directive") or "",
        "prior_evidence": prior_evidence,
        "dimensions": [],
        "sentiment_map": {},
        "insight": "",
    }

    result = _subgraph.invoke(sub_initial)

    # Confidence aggregated from sentiment_map
    confidences = [v["confidence"] for v in result["sentiment_map"].values()]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5

    return {
        "agent_results": [{
            "agent_name": "insight",
            "content": result["insight"],
            "sources": [],  # Insight doesn't cite URLs; it interprets
            "confidence": round(avg_conf, 2),
        }],
        "total_tokens": 800,  # rough: dim extraction + sentiment + synthesis
    }