"""Top-level StateGraph wiring all subgraphs."""
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from .state import OverallState
from .supervisor import supervisor_node, route_to_next


def _stub_agent(name: str):
    """Stub until the real subgraph is wired in."""
    def node(state: OverallState) -> dict:
        directive = state.get("host_directive") or "(no directive)"
        return {
            "agent_results": [{
                "agent_name": name,
                "content": f"[stub {name}] received directive: {directive}",
                "sources": [],
                "confidence": 0.5,
            }],
            "total_tokens": 50,
        }
    return node


def build_graph(checkpoint_path: str = "checkpoints.db"):
    g = StateGraph(OverallState)

    g.add_node("supervisor", supervisor_node)
    for name in ["query", "media", "insight", "devil_advocate", "report"]:
        g.add_node(name, _stub_agent(name))

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_to_next,
        {
            "query": "query",
            "media": "media",
            "insight": "insight",
            "devil_advocate": "devil_advocate",
            "report": "report",
            "__end__": END,
        },
    )
    for agent in ["query", "media", "insight", "devil_advocate"]:
        g.add_edge(agent, "supervisor")
    g.add_edge("report", END)

    # Day 1: use in-memory checkpointer. SQLite persistence added in Phase 3.
    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)