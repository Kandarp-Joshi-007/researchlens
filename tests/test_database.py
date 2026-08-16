"""Database round-trips, migrations and run versioning."""

import sqlite3


class TestRoundTrips:
    def test_paper_and_score(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "Title", "Author", 10,
                                      filepath="/tmp/f.pdf", sha256="abc")
        temp_db.save_score(paper_id, "risk", 3.0, "because",
                           ["point"], ["quote"], run_id=1)
        scores = temp_db.get_paper_scores(paper_id)
        assert len(scores) == 1
        assert scores[0]["key_points"] == ["point"]
        assert scores[0]["evidence"] == ["quote"]

    def test_summary_upsert_does_not_duplicate(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1)
        temp_db.save_summary(paper_id, 5.0, "Moderate")
        temp_db.save_summary(paper_id, 8.0, "Strong")
        assert temp_db.get_paper_summary(paper_id)["overall"] == 8.0

    def test_malformed_json_does_not_crash(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1)
        with temp_db.get_conn() as conn:
            conn.execute(
                "INSERT INTO scores (paper_id, agent, score, rationale,"
                " key_points, evidence) VALUES (?,?,?,?,?,?)",
                (paper_id, "risk", 5.0, "r", "not json", None),
            )
        scores = temp_db.get_paper_scores(paper_id)
        assert scores[0]["key_points"] == [] and scores[0]["evidence"] == []


class TestRunVersioning:
    def test_latest_run_replaces_previous_scores(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1)
        first = temp_db.create_run(paper_id)
        temp_db.save_score(paper_id, "risk", 2.0, "old", [], [], run_id=first)
        second = temp_db.create_run(paper_id)
        temp_db.save_score(paper_id, "risk", 8.0, "new", [], [], run_id=second)

        scores = temp_db.get_paper_scores(paper_id)
        assert len(scores) == 1, "re-analysis must not stack duplicate rows"
        assert scores[0]["score"] == 8.0

    def test_history_keeps_every_run(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1)
        for score in (2.0, 8.0):
            run_id = temp_db.create_run(paper_id)
            temp_db.save_score(paper_id, "risk", score, "r", [], [], run_id=run_id)
        history = temp_db.get_run_history(paper_id)
        assert [h["scores"]["risk"] for h in history] == [2.0, 8.0]

    def test_run_status_transitions(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1)
        run_id = temp_db.create_run(paper_id)
        assert temp_db.get_latest_run(paper_id)["stage"] == "queued"
        temp_db.update_run(run_id, "scoring:risk")
        assert temp_db.get_latest_run(paper_id)["done"] == 0
        temp_db.update_run(run_id, "done", done=True)
        assert temp_db.get_latest_run(paper_id)["done"] == 1

    def test_legacy_rows_without_run_id_still_readable(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1)
        temp_db.save_score(paper_id, "risk", 4.0, "r", [], [], run_id=None)
        assert temp_db.get_paper_scores(paper_id)[0]["score"] == 4.0


class TestDedupe:
    def test_only_matches_completed_analyses(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1, sha256="deadbeef")
        assert temp_db.find_by_hash("deadbeef") is None, "unscored paper is not a cache hit"
        temp_db.save_summary(paper_id, 7.0, "Moderate")
        assert temp_db.find_by_hash("deadbeef")["id"] == paper_id

    def test_unknown_and_empty_hashes(self, temp_db):
        assert temp_db.find_by_hash("nope") is None
        assert temp_db.find_by_hash("") is None
        assert temp_db.find_by_hash(None) is None


class TestMigrations:
    def test_adds_columns_to_an_old_database(self, tmp_path, monkeypatch):
        from backend.core import database

        path = tmp_path / "legacy.db"
        with sqlite3.connect(str(path)) as conn:
            conn.executescript("""
                CREATE TABLE papers (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL, title TEXT, author TEXT,
                    page_count INTEGER, uploaded_at TEXT);
                CREATE TABLE scores (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER, agent TEXT NOT NULL, score REAL,
                    rationale TEXT, key_points TEXT, scored_at TEXT);
                INSERT INTO papers (filename, title) VALUES ('old.pdf', 'Old');
                INSERT INTO scores (paper_id, agent, score, rationale, key_points)
                    VALUES (1, 'risk', 5.0, 'r', '[]');
            """)

        monkeypatch.setattr(database, "DB_PATH", path)
        database.init_db()

        with database.get_conn() as conn:
            columns = {r["name"] for r in conn.execute("PRAGMA table_info(papers)")}
        assert {"filepath", "prior_art", "sha256"} <= columns
        # Pre-existing data survives.
        assert database.get_paper_scores(1)[0]["score"] == 5.0

    def test_init_db_is_idempotent(self, temp_db):
        temp_db.init_db()
        temp_db.init_db()
        assert temp_db.get_all_results() == []


class TestDeletion:
    def test_removes_all_related_rows(self, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1, filepath="/tmp/x.pdf")
        run_id = temp_db.create_run(paper_id)
        temp_db.save_score(paper_id, "risk", 5.0, "r", [], [], run_id=run_id)
        temp_db.save_summary(paper_id, 5.0, "Moderate")

        assert temp_db.delete_paper(paper_id) == "/tmp/x.pdf"
        assert temp_db.get_paper_scores(paper_id) == []
        assert temp_db.get_paper_summary(paper_id) is None
        assert temp_db.get_latest_run(paper_id) is None
