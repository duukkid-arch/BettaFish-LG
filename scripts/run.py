"""CLI entry point."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.panel import Panel
from graph.main_graph import build_graph

console = Console()


@click.command()
@click.argument("topic")
@click.option("--session", default="default", help="Session ID for memory.")
def main(topic: str, session: str):
    """Run sentiment analysis on a topic."""
    console.print(Panel.fit(
        f"[bold cyan]BettaFish-LG v0.1[/]\nTopic: {topic}",
        border_style="cyan",
    ))

    graph = build_graph()
    initial_state = {
        "topic": topic,
        "session_id": session,
        "historical_context": None,
        "agent_results": [],
        "host_directive": None,
        "debate_round": 0,
        "next_speaker": None,
        "should_terminate": False,
        "final_report": None,
        "total_tokens": 0,
    }
    config = {
        "configurable": {"thread_id": session},
        "recursion_limit": 20,
    }

    total_tokens = 0
    for event in graph.stream(initial_state, config=config):
        for node_name, payload in event.items():
            console.print(f"\n[bold yellow]→ {node_name}[/]")
            if payload.get("host_directive"):
                console.print(f"  [dim cyan]directive:[/] {payload['host_directive']}")
            for r in payload.get("agent_results", []):
                console.print(f"  [bold]{r['agent_name']}[/]:")
                console.print(f"    {r['content']}")
                if r.get("sources"):
                    console.print(f"    [dim]sources: {len(r['sources'])} URLs[/]")
            if payload.get("next_speaker"):
                console.print(f"  [dim]→ next:[/] {payload['next_speaker']}")
            total_tokens += payload.get("total_tokens", 0)

    console.print(f"\n[bold green]✓ Done.[/] Total tokens (rough): {total_tokens}")


if __name__ == "__main__":
    main()