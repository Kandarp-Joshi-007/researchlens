import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "results.db"


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                title       TEXT,
                author      TEXT,
                page_count  INTEGER,
                filepath    TEXT,
                uploaded_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scores (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id        INTEGER REFERENCES papers(id),
                agent           TEXT NOT NULL,
                score           REAL,
                rationale       TEXT,
                key_points      TEXT,
                scored_at       TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id    INTEGER REFERENCES papers(id) UNIQUE,
                overall     REAL,
                verdict     TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
        """)
        # Migrate existing DBs that lack the filepath column
        try:
            conn.execute("ALTER TABLE papers ADD COLUMN filepath TEXT")
        except Exception:
            pass


def save_paper(filename: str, title: str, author: str, page_count: int, filepath: str = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO papers (filename, title, author, page_count, filepath) VALUES (?,?,?,?,?)",
            (filename, title, author, page_count, filepath),
        )
        return cur.lastrowid


def delete_paper(paper_id: int) -> Optional[str]:
    """Delete paper and all related data. Returns the stored filepath if any."""
    with get_conn() as conn:
        row = conn.execute("SELECT filepath FROM papers WHERE id=?", (paper_id,)).fetchone()
        filepath = row["filepath"] if row else None
        conn.execute("DELETE FROM summaries WHERE paper_id=?", (paper_id,))
        conn.execute("DELETE FROM scores WHERE paper_id=?", (paper_id,))
        conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
    return filepath


def save_score(paper_id: int, agent: str, score: float, rationale: str, key_points: list):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO scores (paper_id, agent, score, rationale, key_points) VALUES (?,?,?,?,?)",
            (paper_id, agent, score, rationale, json.dumps(key_points)),
        )


def save_summary(paper_id: int, overall: float, verdict: str):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO summaries (paper_id, overall, verdict)
               VALUES (?,?,?)
               ON CONFLICT(paper_id) DO UPDATE SET overall=excluded.overall, verdict=excluded.verdict""",
            (paper_id, overall, verdict),
        )


def get_all_results() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT p.id, p.filename, p.title, p.uploaded_at,
                   s.overall, s.verdict
            FROM papers p
            LEFT JOIN summaries s ON s.paper_id = p.id
            ORDER BY p.uploaded_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_paper_summary(paper_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT overall, verdict FROM summaries WHERE paper_id=?", (paper_id,)
        ).fetchone()
        return dict(row) if row else None


def get_paper_scores(paper_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT agent, score, rationale, key_points, scored_at FROM scores WHERE paper_id=? ORDER BY agent",
            (paper_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["key_points"] = json.loads(d["key_points"] or "[]")
            result.append(d)
        return result
