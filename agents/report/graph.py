"""Report Agent: synthesizes a final Markdown report from all agent contributions."""
import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage

from tools.llm import get_llm
from graph.state import OverallState


class ReportState(TypedDict):
    topic: str
    all_evidence: str
    all_sources: List[str]
    report_md: str
    saved_path: str


# ──────────────────────────────────────────────────────────────────────────
# Node 1: synthesize a 5-section Markdown report
# ──────────────────────────────────────────────────────────────────────────
REPORT_PROMPT = """你是一名资深舆情分析师。请基于以下多 Agent 调研结果，撰写一份结构化舆情分析报告。

分析主题：{topic}

各 Agent 的调研产出：
{evidence}

请生成一份 Markdown 格式的报告，**严格使用以下 5 段结构**：

# {topic} 舆情分析报告

## 1. 执行摘要
（150-200 字，提炼核心结论与最关键的 3-4 个洞察）

## 2. 核心事实与数据
（300-400 字，整合 Query Agent 找到的事实性信息，引用具体数字、benchmark、时间节点）

## 3. 多维度深度分析
（300-400 字，整合 Insight Agent 的深度洞察，按维度展开）

## 4. 媒体生态与叙事
（200-300 字，整合 Media Agent 的媒体格局分析）

## 5. 风险提示与展望
（200-300 字，指出未解之谜、潜在风险、值得追踪的信号）

要求：
- 不重复同一信息——各段聚焦不同层次
- 引用具体数字与事实，不要空泛
- 末尾不要写"总的来说"这种废话
- 不要添加"参考资料"段落（系统会自动追加）
"""


def synthesize_report(state: ReportState) -> dict:
    llm = get_llm(tier="premium", temperature=0.5)  # use qwen-plus for final synthesis
    prompt = REPORT_PROMPT.format(
        topic=state["topic"],
        evidence=state["all_evidence"],
    )
    report_body = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    # Append sources section
    sources_section = ""
    if state["all_sources"]:
        unique_sources = list(dict.fromkeys(state["all_sources"]))  # dedup, keep order
        sources_section = "\n\n## 参考资料\n" + "\n".join(
            f"- {url}" for url in unique_sources[:15]
        )

    report_md = report_body + sources_section
    return {"report_md": report_md}


# ──────────────────────────────────────────────────────────────────────────
# Node 2: save to disk
# ──────────────────────────────────────────────────────────────────────────
def save_report(state: ReportState) -> dict:
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() else "_" for c in state["topic"])[:40]
    filename = f"{timestamp}_{safe_topic}.md"
    filepath = out_dir / filename

    filepath.write_text(state["report_md"], encoding="utf-8")
    return {"saved_path": str(filepath)}


# ──────────────────────────────────────────────────────────────────────────
# Build subgraph
# ──────────────────────────────────────────────────────────────────────────
def build_report_subgraph():
    g = StateGraph(ReportState)
    g.add_node("synthesize_report", synthesize_report)
    g.add_node("save_report", save_report)

    g.add_edge(START, "synthesize_report")
    g.add_edge("synthesize_report", "save_report")
    g.add_edge("save_report", END)

    return g.compile()


# ──────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────
_subgraph = None


def report_agent_node(state: OverallState) -> dict:
    global _subgraph
    if _subgraph is None:
        _subgraph = build_report_subgraph()

    evidence_pieces: List[str] = []
    all_sources: List[str] = []
    for r in state.get("agent_results", []):
        evidence_pieces.append(
            f"=== {r['agent_name'].upper()} Agent ===\n"
            f"置信度: {r.get('confidence', '?')}\n"
            f"{r['content']}"
        )
        all_sources.extend(r.get("sources", []))

    sub_initial: ReportState = {
        "topic": state["topic"],
        "all_evidence": "\n\n".join(evidence_pieces) or "(no evidence collected)",
        "all_sources": all_sources,
        "report_md": "",
        "saved_path": "",
    }

    result = _subgraph.invoke(sub_initial)

    return {
        "agent_results": [{
            "agent_name": "report",
            "content": f"报告已生成：{result['saved_path']}\n\n{result['report_md'][:500]}...（已落盘到文件）",
            "sources": list(dict.fromkeys(all_sources))[:15],
            "confidence": 0.9,
        }],
        "final_report": result["report_md"],
        "total_tokens": 1500,  # qwen-plus is more expensive
    }