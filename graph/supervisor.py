"""Supervisor: routes execution and broadcasts directives."""
import json
from typing import Literal
from langchain_core.messages import SystemMessage
from tools.llm import get_llm
from .state import OverallState

MAX_ROUNDS = 4

DIRECTIVE_PROMPT = """You are the moderator of a multi-agent sentiment analysis.

Topic: {topic}
Round: {round}/{max_rounds}
Historical context: {history}

Recent agent outputs:
{recent}

Decide:
1. The next agent's directive (1-2 sentences, what angle to focus on).
2. Which agent speaks next. Choose ONE of:
   query, media, insight, devil_advocate, report

Output STRICT JSON only, no markdown:
{{"directive": "...", "next_speaker": "..."}}
"""


def supervisor_node(state: OverallState) -> dict:
    round_num = state["debate_round"]

    # Force final synthesis after MAX_ROUNDS
    if round_num >= MAX_ROUNDS:
        return {
            "next_speaker": "report",
            "host_directive": "Synthesize all findings into the final report.",
            "debate_round": round_num + 1,
        }

    llm = get_llm(tier="default", temperature=0.2)

    recent = "\n".join(
        f"[{r['agent_name']}] {r['content'][:300]}"
        for r in state["agent_results"][-3:]
    ) or "(no agent has spoken yet)"

    prompt = DIRECTIVE_PROMPT.format(
        topic=state["topic"],
        round=round_num,
        max_rounds=MAX_ROUNDS,
        history=state.get("historical_context") or "(none)",
        recent=recent,
    )

    resp = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    # Robust JSON parsing
    if resp.startswith("```"):
        # Strip markdown fences if present
        parts = resp.split("```")
        if len(parts) >= 2:
            resp = parts[1].lstrip("json").strip()

    directive = "Continue analysis."
    next_speaker = "query"
    try:
        data = json.loads(resp)
        directive = data.get("directive", directive)
        next_speaker = data.get("next_speaker", next_speaker)
    except json.JSONDecodeError:
        pass

    valid = {"query", "media", "insight", "devil_advocate", "report"}
    if next_speaker not in valid:
        next_speaker = "report"

    return {
        "host_directive": directive,
        "next_speaker": next_speaker,
        "debate_round": round_num + 1,
        "total_tokens": 200,
    }


def route_to_next(state: OverallState) -> Literal[
    "query", "media", "insight", "devil_advocate", "report", "__end__"
]:
    if state.get("should_terminate"):
        return "__end__"
    return state.get("next_speaker") or "query"