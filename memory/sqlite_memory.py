"""Lightweight SQLite-based memory for cross-session report retention."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


DB_PATH = os.getenv("MEMORY_DB", "memory.db")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    topic TEXT NOT NULL,
    session_id TEXT NOT NULL,
    report_md TEXT NOT NULL,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (topic, session_id, created_at)
);

CREATE INDEX IF NOT EXISTS idx_topic_normalized
    ON reports(topic);
"""


def _normalize_topic(topic: str) -> str:
    """Normalize for fuzzy matching: lowercase, strip, collapse whitespace."""
    return " ".join(topic.lower().strip().split())


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Idempotent — safe to call on every startup."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True) if "/" in DB_PATH or "\\" in DB_PATH else None
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def save_report(topic: str, session_id: str, report_md: str, summary: str = "") -> None:
    """Persist a completed report."""
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO reports (topic, session_id, report_md, summary) VALUES (?, ?, ?, ?)",
            (_normalize_topic(topic), session_id, report_md, summary),
        )


def get_latest_report(topic: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the most recent prior report on this topic.

    If session_id is given, prefer same-session history; fall back to any session.
    Returns None if no prior report exists.
    """
    init_db()
    normalized = _normalize_topic(topic)

    with _connect() as conn:
        # Try same-session first
        if session_id:
            row = conn.execute(
                "SELECT topic, session_id, report_md, summary, created_at "
                "FROM reports WHERE topic = ? AND session_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (normalized, session_id),
            ).fetchone()
            if row:
                return dict(row)

        # Fall back to any session
        row = conn.execute(
            "SELECT topic, session_id, report_md, summary, created_at "
            "FROM reports WHERE topic = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (normalized,),
        ).fetchone()
        return dict(row) if row else None


def list_topics() -> list:
    """For inspection: list distinct topics with their latest analysis time."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT topic, COUNT(*) as runs, MAX(created_at) as last_run "
            "FROM reports GROUP BY topic ORDER BY last_run DESC"
        ).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    # Quick CLI inspection
    init_db()
    print(f"DB: {DB_PATH}")
    topics = list_topics()
    if not topics:
        print("(empty)")
    else:
        for t in topics:
            print(f"  {t['topic']:30} runs={t['runs']} last={t['last_run']}")