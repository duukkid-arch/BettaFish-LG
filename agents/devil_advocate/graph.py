"""Devil's Advocate Agent: targeted counter-arguments against prior agents' specific claims."""
import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage

from tools.llm import get_llm
from tools.search import search_web
from graph.state import OverallState


class DevilState(TypedDict):
    topic: str
    directive: str
    prior_evidence: str
    extracted_claims: List[Dict[str, str]]   # [{"claim": "...", "source_agent": "...", "quote": "..."}]
    counter_arguments: List[Dict[str, Any]]  # [{"target_claim": "...", "challenge": "...", "evidence": [...], "strength": float}]
    summary: str


# ──────────────────────────────────────────────────────────────────────────
# Node 1: extract specific claims worth challenging
# ──────────────────────────────────────────────────────────────────────────
CLAIM_EXTRACTION_PROMPT = """You are a critical reviewer identifying the strongest claims to challenge.

Topic: {topic}

Prior agent outputs:
{evidence}

Extract 3-4 SPECIFIC claims that meet ALL these criteria:
1. They are concrete (cite a number, name an entity, or make a causal assertion)
2. They are influential (other parts of the analysis depend on them being true)
3. They have non-obvious weaknesses (would benefit from skeptical scrutiny)

DO NOT extract:
- Trivially true statements (e.g. "DeepSeek V3 has 671B parameters" — verifiable, no debate)
- Vague generalities (e.g. "the model performs well")
- Claims already flagged as uncertain by prior agents

Output STRICT JSON array, nothing else:
[
  {{
    "claim": "the specific assertion (1 sentence, in Chinese)",
    "source_agent": "query / insight / media",
    "why_vulnerable": "1 sentence explaining the weakness (in Chinese)"
  }}
]
"""


def extract_claims(state: DevilState) -> dict:
    llm = get_llm(tier="default", temperature=0.3)
    prompt = CLAIM_EXTRACTION_PROMPT.format(
        topic=state["topic"],
        evidence=state.get("prior_evidence") or "(no prior evidence)",
    )
    resp = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    if resp.startswith("```"):
        parts = resp.split("```")
        if len(parts) >= 2:
            resp = parts[1].lstrip("json").strip()

    claims: List[Dict[str, str]] = []
    try:
        parsed = json.loads(resp)
        if isinstance(parsed, list):
            for item in parsed[:4]:
                if isinstance(item, dict) and "claim" in item:
                    claims.append({
                        "claim": str(item.get("claim", "")),
                        "source_agent": str(item.get("source_agent", "unknown")),
                        "why_vulnerable": str(item.get("why_vulnerable", "")),
                    })
    except json.JSONDecodeError:
        pass

    return {"extracted_claims": claims}


# ──────────────────────────────────────────────────────────────────────────
# Node 2: for each claim, search for counter-evidence and construct a challenge
# ──────────────────────────────────────────────────────────────────────────
CHALLENGE_PROMPT = """You are mounting a specific counter-argument.

Topic: {topic}
Target claim (from {source_agent} agent):
"{claim}"

Why this claim is potentially vulnerable: {why_vulnerable}

Counter-evidence from independent searches:
{counter_evidence}

Construct your challenge as STRICT JSON:
{{
  "challenge": "2-3 Chinese sentences directly questioning the claim. Be specific, cite the counter-evidence.",
  "strength": <0.0-1.0, how strong is your counter-case>,
  "concession": "if the counter-evidence is weak and the original claim mostly holds, state that honestly here in 1 sentence. Otherwise empty string."
}}

If counter-evidence is empty or irrelevant, your strength should be LOW (≤0.3) and concession should acknowledge the original claim likely stands.
"""


def construct_challenges(state: DevilState) -> dict:
    challenges: List[Dict[str, Any]] = []
    llm = get_llm(tier="default", temperature=0.4)

    for claim_obj in state["extracted_claims"]:
        # Search for counter-evidence using the claim itself as query
        # Add "质疑" or "争议" to bias toward critical sources
        search_query = f"{claim_obj['claim']} 争议 质疑"
        results = search_web(search_query, max_results=3)

        counter_evidence = "\n".join(
            f"- {r['title']}: {r['content'][:200]}"
            for r in results
        ) or "(no counter-evidence found)"

        prompt = CHALLENGE_PROMPT.format(
            topic=state["topic"],
            source_agent=claim_obj["source_agent"],
            claim=claim_obj["claim"],
            why_vulnerable=claim_obj["why_vulnerable"],
            counter_evidence=counter_evidence,
        )

        resp = llm.invoke([SystemMessage(content=prompt)]).content.strip()
        if resp.startswith("```"):
            parts = resp.split("```")
            if len(parts) >= 2:
                resp = parts[1].lstrip("json").strip()

        try:
            parsed = json.loads(resp)
            challenges.append({
                "target_claim": claim_obj["claim"],
                "source_agent": claim_obj["source_agent"],
                "challenge": str(parsed.get("challenge", "")),
                "strength": float(parsed.get("strength", 0.3)),
                "concession": str(parsed.get("concession", "")),
                "evidence_count": len(results),
            })
        except (json.JSONDecodeError, ValueError, TypeError):
            challenges.append({
                "target_claim": claim_obj["claim"],
                "source_agent": claim_obj["source_agent"],
                "challenge": "(challenge construction failed)",
                "strength": 0.0,
                "concession": "",
                "evidence_count": 0,
            })

    return {"counter_arguments": challenges}


# ──────────────────────────────────────────────────────────────────────────
# Node 3: synthesize into a single Devil's Advocate summary
# ──────────────────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = """You are the Devil's Advocate. Compile your counter-arguments into a focused critique.

Topic: {topic}
Host directive: {directive}

Your challenges (with confidence scores):
{challenges_formatted}

Write a 250-350 word Chinese critique that:
1. Opens with your STRONGEST challenge (highest strength score)
2. Bundles related challenges if they attack the same underlying assumption
3. Honestly notes where your counter-evidence is weak (use the concessions)
4. Ends with ONE specific question the report should not avoid answering

Do NOT:
- Repeat the original claims at length (the reader already saw them)
- Sound like an essay; sound like a skeptical reviewer's note
- Manufacture criticism where you found no counter-evidence
"""


def synthesize_critique(state: DevilState) -> dict:
    if not state["counter_arguments"]:
        return {
            "summary": "Devil's Advocate 未能从前序分析中识别可挑战的高影响力论点，可能因证据已足够审慎。",
        }

    # Sort by strength descending so strongest challenge appears first
    sorted_challenges = sorted(
        state["counter_arguments"],
        key=lambda c: c["strength"],
        reverse=True,
    )

    challenges_text = "\n\n".join(
        f"[强度 {c['strength']:.2f}] 针对 {c['source_agent']} 的论点「{c['target_claim'][:80]}」\n"
        f"质疑：{c['challenge']}\n"
        f"让步：{c['concession'] or '(无)'}"
        for c in sorted_challenges
    )

    llm = get_llm(tier="default", temperature=0.5)
    prompt = SYNTHESIS_PROMPT.format(
        topic=state["topic"],
        directive=state.get("directive") or "Surface counter-evidence and challenge consensus.",
        challenges_formatted=challenges_text,
    )
    summary = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    return {"summary": summary}


# ──────────────────────────────────────────────────────────────────────────
# Build subgraph
# ──────────────────────────────────────────────────────────────────────────
def build_devil_subgraph():
    g = StateGraph(DevilState)
    g.add_node("extract_claims", extract_claims)
    g.add_node("construct_challenges", construct_challenges)
    g.add_node("synthesize_critique", synthesize_critique)

    g.add_edge(START, "extract_claims")
    g.add_edge("extract_claims", "construct_challenges")
    g.add_edge("construct_challenges", "synthesize_critique")
    g.add_edge("synthesize_critique", END)

    return g.compile()


# ──────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────
_subgraph = None


def devil_advocate_node(state: OverallState) -> dict:
    """Entry point called by the top-level graph."""
    global _subgraph
    if _subgraph is None:
        _subgraph = build_devil_subgraph()

    # Only meaningful if prior agents have produced substantive output
    prior_results = state.get("agent_results", [])
    if not prior_results:
        return {
            "agent_results": [{
                "agent_name": "devil_advocate",
                "content": "(无前序输出可挑战，跳过)",
                "sources": [],
                "confidence": 0.0,
            }],
            "total_tokens": 50,
        }

    evidence = "\n\n".join(
        f"=== {r['agent_name'].upper()} ===\n{r['content']}"
        for r in prior_results
    )

    sub_initial: DevilState = {
        "topic": state["topic"],
        "directive": state.get("host_directive") or "",
        "prior_evidence": evidence,
        "extracted_claims": [],
        "counter_arguments": [],
        "summary": "",
    }

    result = _subgraph.invoke(sub_initial)

    # Average challenge strength as confidence signal
    challenges = result.get("counter_arguments", [])
    if challenges:
        avg_strength = sum(c["strength"] for c in challenges) / len(challenges)
    else:
        avg_strength = 0.0

    return {
        "agent_results": [{
            "agent_name": "devil_advocate",
            "content": result["summary"],
            "sources": [],  # challenges may cite evidence inline but no clean URL list
            "confidence": round(avg_strength, 2),
        }],
        "total_tokens": 1200,  # 4 claims × ~250 tokens each + synthesis
    }