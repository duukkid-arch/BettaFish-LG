"""Report Agent: synthesizes a final Markdown report from all agent contributions."""
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage

from tools.llm import get_llm
from memory.sqlite_memory import save_report, get_latest_report
from graph.state import OverallState


class ReportState(TypedDict):
    topic: str
    session_id: str
    all_evidence: str
    all_sources: List[str]
    prior_report_md: Optional[str]      # for diff
    report_md: str
    summary: str
    saved_path: str


# ──────────────────────────────────────────────────────────────────────────
# Node 1: synthesize the main 5-section report
# ──────────────────────────────────────────────────────────────────────────
REPORT_PROMPT = """你是一名资深舆情分析师。请基于以下多 Agent 调研结果，撰写一份结构化舆情分析报告。

分析主题：{topic}

各 Agent 的调研产出：
{evidence}

请生成一份 Markdown 格式的报告，**严格使用以下 5 段结构**：

# {topic} 舆情分析报告

## 1. 执行摘要
（150-200 字，提炼核心结论与最关键的 3-4 个洞察。如果有 Devil's Advocate 的对抗性观点，必须吸收。）

## 2. 核心事实与数据
（300-400 字，整合 Query Agent 找到的事实性信息，引用具体数字、benchmark、时间节点）

## 3. 多维度深度分析
（300-400 字，整合 Insight Agent 的深度洞察，按维度展开）

## 4. 媒体生态与叙事
（200-300 字，整合 Media Agent 的媒体格局分析）

## 5. 风险提示与展望
（200-300 字，融入 Devil's Advocate 提出的关键质疑，给出可追踪的信号）

要求：
- 不重复同一信息——各段聚焦不同层次
- 引用具体数字与事实，不要空泛
- 末尾不要写"总的来说"这种废话
- 不要添加"参考资料"段落（系统会自动追加）
"""


def synthesize_report(state: ReportState) -> dict:
    llm = get_llm(tier="premium", temperature=0.5)
    prompt = REPORT_PROMPT.format(topic=state["topic"], evidence=state["all_evidence"])
    report_body = llm.invoke([SystemMessage(content=prompt)]).content.strip()
    return {"report_md": report_body}


# ──────────────────────────────────────────────────────────────────────────
# Node 2: if a prior report exists, append an evolution diff section
# ──────────────────────────────────────────────────────────────────────────
DIFF_PROMPT = """你正在为舆情演变追踪生成对比段落。

主题：{topic}

【上一次的报告】（早先生成）：
{prior}

【本次的报告】（刚生成）：
{current}

请生成一段 Markdown 内容，标题为 "## 6. 自上次分析以来的变化"，必须按以下结构：

## 6. 自上次分析以来的变化

**【新增】**
- 列出本次出现但上次没有的关键事实/判断/数据（每条 1 句话）

**【反转】**
- 列出与上次结论方向相反的判断（每条 1 句话，说明为什么反转）

**【失效/淘汰】**
- 列出上次提到但本次不再相关的信号（每条 1 句话）

**【持续】**
- 列出两次都强调、仍然成立的核心判断（不超过 2 条）

要求：
- 客观对比，不要发明上次没有的内容
- 如果某一类（新增/反转/失效）确实没有，写 "无显著变化"
- 不要超过 250 字
"""


def append_evolution_diff(state: ReportState) -> dict:
    prior_md = state.get("prior_report_md")
    if not prior_md:
        # First-ever run, no diff to append
        return {}

    llm = get_llm(tier="default", temperature=0.3)
    prompt = DIFF_PROMPT.format(
        topic=state["topic"],
        prior=prior_md[:4000],  # truncate to control tokens
        current=state["report_md"][:4000],
    )
    diff_section = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    return {
        "report_md": state["report_md"] + "\n\n" + diff_section,
    }


# ──────────────────────────────────────────────────────────────────────────
# Node 3: produce a one-sentence summary for memory
# ──────────────────────────────────────────────────────────────────────────
SUMMARY_PROMPT = """以下是一份舆情分析报告。请用 ONE 中文句子（≤50 字）提炼其最核心结论，要包含核心判断与关键依据。

报告：
{report}

直接输出那一句话，不要任何前缀（如"总结："）。
"""


def extract_summary(state: ReportState) -> dict:
    llm = get_llm(tier="default", temperature=0.2)
    prompt = SUMMARY_PROMPT.format(report=state["report_md"][:3000])
    summary = llm.invoke([SystemMessage(content=prompt)]).content.strip()
    # Strip quotes if model wrapped it
    summary = summary.strip('"\'""''')
    return {"summary": summary}


# ──────────────────────────────────────────────────────────────────────────
# Node 4: append sources + save to disk + persist to SQLite
# ──────────────────────────────────────────────────────────────────────────
def finalize_and_save(state: ReportState) -> dict:
    # Append sources section
    sources_section = ""
    if state["all_sources"]:
        unique_sources = list(dict.fromkeys(state["all_sources"]))
        sources_section = "\n\n## 参考资料\n" + "\n".join(
            f"- {url}" for url in unique_sources[:15]
        )

    final_md = state["report_md"] + sources_section

    # Save to disk
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() else "_" for c in state["topic"])[:40]
    filepath = out_dir / f"{timestamp}_{safe_topic}.md"
    filepath.write_text(final_md, encoding="utf-8")

    # Persist to SQLite for future evolution tracking
    try:
        save_report(
            topic=state["topic"],
            session_id=state.get("session_id", "default"),
            report_md=final_md,
            summary=state.get("summary", ""),
        )
    except Exception as e:
        # Memory failure must not break the run
        print(f"[memory] save failed: {type(e).__name__}: {e}")

    return {"report_md": final_md, "saved_path": str(filepath)}


# ──────────────────────────────────────────────────────────────────────────
# Build subgraph
# ──────────────────────────────────────────────────────────────────────────
def build_report_subgraph():
    g = StateGraph(ReportState)
    g.add_node("synthesize_report", synthesize_report)
    g.add_node("append_evolution_diff", append_evolution_diff)
    g.add_node("extract_summary", extract_summary)
    g.add_node("finalize_and_save", finalize_and_save)

    g.add_edge(START, "synthesize_report")
    g.add_edge("synthesize_report", "append_evolution_diff")
    g.add_edge("append_evolution_diff", "extract_summary")
    g.add_edge("extract_summary", "finalize_and_save")
    g.add_edge("finalize_and_save", END)

    return g.compile()


# ──────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────
_subgraph = None


def report_agent_node(state: OverallState) -> dict:
    global _subgraph
    if _subgraph is None:
        _subgraph = build_report_subgraph()

    # Collect all evidence
    evidence_pieces: List[str] = []
    all_sources: List[str] = []
    for r in state.get("agent_results", []):
        evidence_pieces.append(
            f"=== {r['agent_name'].upper()} Agent ===\n"
            f"置信度: {r.get('confidence', '?')}\n"
            f"{r['content']}"
        )
        all_sources.extend(r.get("sources", []))

    # Look up prior report for diff
    prior = get_latest_report(state["topic"], state.get("session_id"))
    prior_md = prior["report_md"] if prior else None

    sub_initial: ReportState = {
        "topic": state["topic"],
        "session_id": state.get("session_id", "default"),
        "all_evidence": "\n\n".join(evidence_pieces) or "(no evidence collected)",
        "all_sources": all_sources,
        "prior_report_md": prior_md,
        "report_md": "",
        "summary": "",
        "saved_path": "",
    }

    result = _subgraph.invoke(sub_initial)

    return {
        "agent_results": [{
            "agent_name": "report",
            "content": f"报告已生成：{result['saved_path']}\n\n{result['report_md'][:500]}...（已落盘并保存到记忆）",
            "sources": list(dict.fromkeys(all_sources))[:15],
            "confidence": 0.9,
        }],
        "final_report": result["report_md"],
        "total_tokens": 2000,  # qwen-plus synthesis + diff + summary
    }