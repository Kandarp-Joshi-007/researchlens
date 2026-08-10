import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "results.db"


@contextmanager
def get_conn():
    """Commit-and-close connection scope.

    `with sqlite3.connect(...)` commits but never closes, so connections used to
    linger until garbage collection. WAL plus a busy timeout lets the background
    analysis write while the API reads instead of raising "database is locked".
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        with conn:
            yield conn
    finally:
        conn.close()


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
                evidence        TEXT,
                scored_at       TEXT DEFAULT (datetime('now'))
            );

            -- One analysis attempt. Survives a backend restart, so a paper
            -- mid-run is resumable rather than stuck forever.
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id    INTEGER REFERENCES papers(id),
                stage       TEXT,
                done        INTEGER DEFAULT 0,
                error       TEXT,
                started_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id    INTEGER REFERENCES papers(id) UNIQUE,
                overall     REAL,
                verdict     TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
        """)
        _add_column(conn, "papers", "filepath", "TEXT")
        _add_column(conn, "papers", "prior_art", "TEXT")
        _add_column(conn, "papers", "sha256", "TEXT")
        _add_column(conn, "scores", "evidence", "TEXT")
        _add_column(conn, "scores", "run_id", "INTEGER")
        _add_column(conn, "scores", "score_min", "REAL")
        _add_column(conn, "scores", "score_max", "REAL")
        _add_column(conn, "scores", "samples", "INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_sha ON papers(sha256)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_paper ON scores(paper_id, run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_paper ON runs(paper_id)")


def _add_column(conn, table: str, column: str, coltype: str):
    """Add a column to an existing database, ignoring the already-present case."""
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def save_paper(filename: str, title: str, author: str, page_count: int,
               filepath: str = None, sha256: str = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO papers (filename, title, author, page_count, filepath, sha256)"
            " VALUES (?,?,?,?,?,?)",
            (filename, title, author, page_count, filepath, sha256),
        )
        return cur.lastrowid


def get_paper(paper_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, filename, title, author, page_count, filepath, uploaded_at"
            " FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
    return dict(row) if row else None


def find_by_hash(sha256: str) -> Optional[dict]:
    """An already-analysed paper with identical content, if one exists.

    Only rows with a stored summary count: a half-finished or failed run
    should be redone rather than served from cache.
    """
    if not sha256:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT p.id, p.title, p.page_count, s.overall, s.verdict"
            " FROM papers p JOIN summaries s ON s.paper_id = p.id"
            " WHERE p.sha256 = ? ORDER BY p.id DESC LIMIT 1",
            (sha256,),
        ).fetchone()
    return dict(row) if row else None


def delete_paper(paper_id: int) -> Optional[str]:
    """Delete paper and all related data. Returns the stored filepath if any."""
    with get_conn() as conn:
        row = conn.execute("SELECT filepath FROM papers WHERE id=?", (paper_id,)).fetchone()
        filepath = row["filepath"] if row else None
        conn.execute("DELETE FROM summaries WHERE paper_id=?", (paper_id,))
        conn.execute("DELETE FROM scores WHERE paper_id=?", (paper_id,))
        conn.execute("DELETE FROM runs WHERE paper_id=?", (paper_id,))
        conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
    return filepath


def create_run(paper_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs (paper_id, stage, done) VALUES (?, 'queued', 0)",
            (paper_id,),
        )
        return cur.lastrowid


def update_run(run_id: int, stage: str, done: bool = False, error: str = None):
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET stage=?, done=?, error=?, updated_at=datetime('now')"
            " WHERE id=?",
            (stage, 1 if done else 0, error, run_id),
        )


def get_latest_run(paper_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, stage, done, error, started_at, updated_at FROM runs"
            " WHERE paper_id=? ORDER BY id DESC LIMIT 1",
            (paper_id,),
        ).fetchone()
    return dict(row) if row else None


def get_run_history(paper_id: int) -> list:
    """Every completed run's scores, oldest first — how a paper's view changed."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT run_id, agent, score, scored_at FROM scores"
            " WHERE paper_id=? AND run_id IS NOT NULL ORDER BY run_id, agent",
            (paper_id,),
        ).fetchall()
    runs = {}
    for row in rows:
        runs.setdefault(row["run_id"], {"run_id": row["run_id"], "scores": {},
                                        "scored_at": row["scored_at"]})
        runs[row["run_id"]]["scores"][row["agent"]] = row["score"]
    return [runs[k] for k in sorted(runs)]


def save_score(paper_id: int, agent: str, score: float, rationale: str,
               key_points: list, evidence: list = None, run_id: int = None,
               score_min: float = None, score_max: float = None,
               samples: int = 1):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO scores (paper_id, agent, score, rationale, key_points,"
            " evidence, run_id, score_min, score_max, samples)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (paper_id, agent, score, rationale, json.dumps(key_points),
             json.dumps(evidence or []), run_id,
             score_min if score_min is not None else score,
             score_max if score_max is not None else score, samples),
        )


def save_summary(paper_id: int, overall: float, verdict: str):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO summaries (paper_id, overall, verdict)
               VALUES (?,?,?)
               ON CONFLICT(paper_id) DO UPDATE SET overall=excluded.overall, verdict=excluded.verdict""",
            (paper_id, overall, verdict),
        )


def save_prior_art(paper_id: int, works: list):
    with get_conn() as conn:
        conn.execute("UPDATE papers SET prior_art=? WHERE id=?",
                     (json.dumps(works or []), paper_id))


def get_prior_art(paper_id: int) -> list:
    with get_conn() as conn:
        row = conn.execute("SELECT prior_art FROM papers WHERE id=?", (paper_id,)).fetchone()
    return _load_json_list(row["prior_art"]) if row else []


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
        # Only the most recent run, so re-analysis replaces rather than
        # duplicates. Rows predating run tracking have run_id NULL and are
        # only used when there is no newer run.
        latest = conn.execute(
            "SELECT MAX(run_id) AS r FROM scores WHERE paper_id=?", (paper_id,)
        ).fetchone()["r"]
        if latest is None:
            rows = conn.execute(
                "SELECT agent, score, rationale, key_points, evidence, scored_at,"
                " score_min, score_max, samples"
                " FROM scores WHERE paper_id=? ORDER BY agent",
                (paper_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT agent, score, rationale, key_points, evidence, scored_at,"
                " score_min, score_max, samples"
                " FROM scores WHERE paper_id=? AND run_id=? ORDER BY agent",
                (paper_id, latest),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["key_points"] = _load_json_list(d.get("key_points"))
            d["evidence"] = _load_json_list(d.get("evidence"))
            result.append(d)
        return result


def _load_json_list(raw) -> list:
    """Parse a stored JSON list, tolerating nulls and pre-migration rows."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except (ValueError, TypeError):
        return []
