"""Top-level StateGraph wiring all subgraphs."""
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from .state import OverallState
from .supervisor import supervisor_node, route_to_next
from agents.query.graph import query_agent_node
from agents.insight.graph import insight_agent_node
from agents.media.graph import media_agent_node
from agents.report.graph import report_agent_node
from agents.devil_advocate.graph import devil_advocate_node


def build_graph(checkpoint_path: str = "checkpoints.db"):
    g = StateGraph(OverallState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("query", query_agent_node)
    g.add_node("insight", insight_agent_node)
    g.add_node("media", media_agent_node)
    g.add_node("devil_advocate", devil_advocate_node)  # ← real now
    g.add_node("report", report_agent_node)

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

    checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)