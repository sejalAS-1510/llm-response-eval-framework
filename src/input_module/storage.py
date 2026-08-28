"""
Very small SQLite wrapper for storing evaluation submissions.
Kept simple on purpose for M1 - just enough to persist what comes in
through the API so M2 can pull it back out for the agent layer.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "submissions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    reference_answer TEXT,
    source_document TEXT,
    created_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def insert_submission(
    question: str,
    ai_response: str,
    reference_answer: Optional[str] = None,
    source_document: Optional[str] = None,
) -> sqlite3.Row:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO submissions (question, ai_response, reference_answer, source_document, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (question, ai_response, reference_answer, source_document, created_at),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM submissions WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return row


def get_submission(submission_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()
