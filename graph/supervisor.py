"""Supervisor: routes execution and broadcasts directives."""
import json
from typing import Literal
from langchain_core.messages import SystemMessage
from tools.llm import get_llm
from memory.sqlite_memory import get_latest_report
from .state import OverallState

MAX_ROUNDS = 5

DIRECTIVE_PROMPT = """You are the moderator of a multi-agent sentiment analysis debate.

Topic: {topic}
Round: {round}/{max_rounds}
Historical context (from prior runs of this topic): {history}

Agents who have already spoken: {spoken_agents}

Recent agent outputs (last 3):
{recent}

Routing rules:
1. Rounds 1-2: prefer evidence gathering (query, media, insight)
2. By round 3, if ANY of query/insight has spoken, you MUST call devil_advocate to surface counter-evidence BEFORE report.
3. Only route to "report" when devil_advocate has had its turn OR round >= {max_rounds}.
4. Don't call the same agent twice in a row.
5. If historical context exists, the directive should reference what to verify vs. last time.

Choose the next agent from: query, media, insight, devil_advocate, report

Output STRICT JSON only, no markdown:
{{"directive": "1-2 sentence focus for the next agent", "next_speaker": "agent_name"}}
"""


def supervisor_node(state: OverallState) -> dict:
    round_num = state["debate_round"]

    # ── On the very first round, inject historical context if available ──
    extra_updates = {}
    if round_num == 0 and not state.get("historical_context"):
        prior = get_latest_report(state["topic"], state.get("session_id"))
        if prior:
            summary = prior.get("summary") or "(no summary available)"
            history_str = (
                f"Last analyzed on {prior['created_at']}. "
                f"Previous conclusion: {summary}"
            )
            extra_updates["historical_context"] = history_str

    # Effective history for this round (just-injected OR pre-existing)
    effective_history = (
        extra_updates.get("historical_context")
        or state.get("historical_context")
        or "(no prior runs)"
    )

    # Hard cap
    if round_num >= MAX_ROUNDS:
        return {
            **extra_updates,
            "next_speaker": "report",
            "host_directive": "Synthesize all findings, including counter-arguments, into the final report.",
            "debate_round": round_num + 1,
        }

    llm = get_llm(tier="default", temperature=0.2)

    spoken_agents = list({r["agent_name"] for r in state.get("agent_results", [])})

    recent = "\n".join(
        f"[{r['agent_name']}] {r['content'][:300]}"
        for r in state["agent_results"][-3:]
    ) or "(no agent has spoken yet)"

    prompt = DIRECTIVE_PROMPT.format(
        topic=state["topic"],
        round=round_num,
        max_rounds=MAX_ROUNDS,
        history=effective_history,
        spoken_agents=", ".join(spoken_agents) or "(none yet)",
        recent=recent,
    )

    resp = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    if resp.startswith("```"):
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

    # Hard constraint: enforce devil_advocate before report
    if (round_num >= 3
        and "devil_advocate" not in spoken_agents
        and next_speaker == "report"):
        next_speaker = "devil_advocate"
        directive = "Surface counter-evidence against the strongest claims made so far."

    return {
        **extra_updates,
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