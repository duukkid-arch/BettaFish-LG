"""Shared state for the top-level graph."""
from typing import TypedDict, Annotated, List, Optional
from operator import add


class AgentResult(TypedDict):
    agent_name: str
    content: str
    sources: List[str]
    confidence: float


class OverallState(TypedDict):
    # Input
    topic: str
    session_id: str

    # Memory injected at start
    historical_context: Optional[str]

    # Per-agent outputs (append-only)
    agent_results: Annotated[List[AgentResult], add]

    # Forum / debate state
    host_directive: Optional[str]
    debate_round: int
    next_speaker: Optional[str]
    should_terminate: bool

    # Output
    final_report: Optional[str]

    # Metrics
    total_tokens: Annotated[int, add]