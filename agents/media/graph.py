"""Media Agent: analyzes media-side characteristics (sources, propagation, narrative angles)."""
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage

from tools.llm import get_llm
from graph.state import OverallState


class MediaState(TypedDict):
    topic: str
    directive: str
    prior_sources: List[str]      # URLs collected by Query
    prior_evidence: str
    media_profile: Dict[str, Any]  # source diversity, narrative angles, etc.
    media_summary: str


# ──────────────────────────────────────────────────────────────────────────
# Node 1: profile the media landscape from collected sources
# ──────────────────────────────────────────────────────────────────────────
PROFILE_PROMPT = """You are a media landscape analyst.

Topic: {topic}
Source URLs collected so far:
{urls}

Evidence content:
{evidence}

Profile the media landscape:
1. source_diversity: how varied are the sources? ("high" / "medium" / "low")
2. dominant_outlets: 2-4 outlets that appear most prominent (domain names in Chinese context if any)
3. narrative_angles: 2-3 distinct angles being pushed (e.g. "技术突破叙事", "商业竞争叙事")
4. tone_balance: overall tone leaning ("celebratory", "critical", "neutral", "mixed")
5. echo_chamber_risk: are the sources reinforcing each other or genuinely independent? ("low" / "medium" / "high")

Output STRICT JSON, nothing else:
{{
  "source_diversity": "...",
  "dominant_outlets": ["...", "..."],
  "narrative_angles": ["...", "..."],
  "tone_balance": "...",
  "echo_chamber_risk": "..."
}}
"""


def profile_media(state: MediaState) -> dict:
    urls_text = "\n".join(f"- {u}" for u in state["prior_sources"][:15]) or "(none)"

    llm = get_llm(tier="default", temperature=0.3)
    prompt = PROFILE_PROMPT.format(
        topic=state["topic"],
        urls=urls_text,
        evidence=state.get("prior_evidence") or "(no evidence)",
    )
    resp = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    if resp.startswith("```"):
        parts = resp.split("```")
        if len(parts) >= 2:
            resp = parts[1].lstrip("json").strip()

    profile: Dict[str, Any] = {}
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, dict):
            profile = parsed
    except json.JSONDecodeError:
        pass

    if not profile:
        profile = {
            "source_diversity": "unknown",
            "dominant_outlets": [],
            "narrative_angles": [],
            "tone_balance": "unknown",
            "echo_chamber_risk": "unknown",
        }

    return {"media_profile": profile}


# ──────────────────────────────────────────────────────────────────────────
# Node 2: synthesize media-side summary
# ──────────────────────────────────────────────────────────────────────────
MEDIA_SUMMARY_PROMPT = """You are a media analyst writing for a sentiment analysis report.

Topic: {topic}
Host directive: {directive}

Media profile:
- 信源多样性: {diversity}
- 主导媒体: {outlets}
- 叙事角度: {angles}
- 整体基调: {tone}
- 回音壁风险: {echo}

Write a 200-280 word Chinese summary that:
1. Describes WHO is shaping the narrative (which outlets / angles dominate)
2. Identifies whether the discourse is genuinely diverse or echo-chamber-driven
3. Flags any narrative angle that the prior agents may have underweighted
4. Does NOT repeat the underlying facts about {topic} itself — focus on the media-LEVEL pattern

Be specific. Avoid generic phrases like "媒体广泛报道".
"""


def synthesize_media(state: MediaState) -> dict:
    p = state["media_profile"]

    llm = get_llm(tier="default", temperature=0.5)
    prompt = MEDIA_SUMMARY_PROMPT.format(
        topic=state["topic"],
        directive=state.get("directive") or "Analyze the media landscape.",
        diversity=p.get("source_diversity", "?"),
        outlets=", ".join(p.get("dominant_outlets", [])) or "未识别",
        angles=", ".join(p.get("narrative_angles", [])) or "未识别",
        tone=p.get("tone_balance", "?"),
        echo=p.get("echo_chamber_risk", "?"),
    )
    summary = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    return {"media_summary": summary}


# ──────────────────────────────────────────────────────────────────────────
# Build the subgraph
# ──────────────────────────────────────────────────────────────────────────
def build_media_subgraph():
    g = StateGraph(MediaState)
    g.add_node("profile_media", profile_media)
    g.add_node("synthesize_media", synthesize_media)

    g.add_edge(START, "profile_media")
    g.add_edge("profile_media", "synthesize_media")
    g.add_edge("synthesize_media", END)

    return g.compile()


# ──────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────
_subgraph = None


def media_agent_node(state: OverallState) -> dict:
    global _subgraph
    if _subgraph is None:
        _subgraph = build_media_subgraph()

    # Gather URLs and evidence from prior agents
    prior_sources: List[str] = []
    evidence_pieces: List[str] = []
    for r in state.get("agent_results", []):
        prior_sources.extend(r.get("sources", []))
        evidence_pieces.append(f"[{r['agent_name']}]\n{r['content']}")

    sub_initial: MediaState = {
        "topic": state["topic"],
        "directive": state.get("host_directive") or "",
        "prior_sources": prior_sources,
        "prior_evidence": "\n\n".join(evidence_pieces) or "(none)",
        "media_profile": {},
        "media_summary": "",
    }

    result = _subgraph.invoke(sub_initial)

    # Confidence: high if we had real sources to profile, low otherwise
    conf = 0.7 if prior_sources else 0.3

    return {
        "agent_results": [{
            "agent_name": "media",
            "content": result["media_summary"],
            "sources": prior_sources[:10],  # carry forward the URLs for report citation
            "confidence": conf,
        }],
        "total_tokens": 500,
    }