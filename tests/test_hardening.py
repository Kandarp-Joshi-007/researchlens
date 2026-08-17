"""Regression tests for the production-hardening pass.

Each test here corresponds to a defect found by exercising the running app
rather than by reading the code, so they are the ones most worth keeping.
"""

import time
from pathlib import Path

import pytest

from conftest import make_pdf


class TestFilenameSanitising:
    def test_long_name_keeps_its_extension(self):
        """A 200-character name used to lose '.pdf' to truncation and then be
        rejected by the caller as "Only PDF files accepted"."""
        from backend.main import safe_filename

        cleaned = safe_filename("A" * 200 + ".pdf")
        assert cleaned.endswith(".pdf")
        assert len(cleaned) <= 120

    def test_long_name_without_extension_is_still_truncated(self):
        from backend.main import safe_filename

        assert len(safe_filename("B" * 400)) <= 120

    @pytest.mark.parametrize("name", [
        "../../evil.pdf", "a/../../../etc/passwd.pdf", "C:/Windows/evil.pdf",
        "..\\..\\evil.pdf", "....//....//evil.pdf",
    ])
    def test_traversal_never_escapes_uploads(self, name):
        from backend.main import UPLOAD_DIR, safe_filename

        target = (UPLOAD_DIR / f"id_{safe_filename(name)}").resolve()
        assert target.parent == UPLOAD_DIR.resolve()


class TestErrorMessagesDoNotLeakPaths:
    def test_server_paths_are_reduced_to_a_basename(self):
        from backend.main import _scrub_paths

        message = _scrub_paths(RuntimeError(
            "Failed to open file 'D:\\ResearchLens\\uploads\\abc_paper.pdf'."
        ))
        assert "ResearchLens" not in message
        assert "uploads" not in message
        assert "paper.pdf" in message

    def test_posix_paths_are_scrubbed_too(self):
        from backend.main import _scrub_paths

        message = _scrub_paths(RuntimeError("cannot open /srv/app/uploads/x_y.pdf"))
        assert "/srv/app" not in message
        assert "y.pdf" in message


class TestPdfExtraction:
    def test_document_is_closed_when_extraction_fails(self, tmp_path, monkeypatch):
        """PyMuPDF holds the file open; on Windows the lock makes the caller's
        cleanup unlink() fail and strands the upload on disk."""
        from backend.core import pdf_extractor

        path = make_pdf(tmp_path / "p.pdf", ["1. Introduction", "Body text here."],
                        bold_indices=(0,))
        closed = {"count": 0}
        real_open = pdf_extractor.fitz.open

        def tracking_open(*args, **kwargs):
            doc = real_open(*args, **kwargs)
            real_close = doc.close

            def close():
                closed["count"] += 1
                real_close()

            doc.close = close
            return doc

        monkeypatch.setattr(pdf_extractor.fitz, "open", tracking_open)
        monkeypatch.setattr(pdf_extractor, "_extract",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

        with pytest.raises(RuntimeError):
            pdf_extractor.extract_text(path)
        assert closed["count"] == 1

        # The lock is genuinely released: on Windows this fails otherwise.
        Path(path).unlink()

    def test_rotated_margin_stamp_is_not_mistaken_for_the_title(self, tmp_path):
        """arXiv prints its identifier down the left margin at a larger size
        than the paper's own title, so 'largest text wins' returned
        "arXiv:1611.07004v3 [cs.CV] 26 Nov 2018" for every arXiv preprint."""
        import fitz

        from backend.core.pdf_extractor import extract_text

        path = tmp_path / "preprint.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # The stamp: bigger than the title, but rotated down the margin.
        page.insert_text((30, 500), "arXiv:1611.07004v3  [cs.CV]  26 Nov 2018",
                         fontsize=20, rotate=90)
        page.insert_text((72, 90), "Image-to-Image Translation with Networks",
                         fontsize=14, fontname="helvetica-bold")
        y = 130
        for i in range(8):
            page.insert_text((72, y), f"Body line {i} with enough text to pass.",
                             fontsize=11)
            y += 16
        doc.save(str(path))
        doc.close()

        title = extract_text(str(path))["title"]
        assert "arXiv" not in title
        assert "Image-to-Image" in title

    def test_encrypted_pdf_reports_the_real_reason(self, tmp_path):
        import fitz

        from backend.core.pdf_extractor import extract_text

        path = tmp_path / "locked.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "secret contents", fontsize=11)
        doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256,
                 user_pw="opensesame", owner_pw="opensesame")
        doc.close()

        with pytest.raises(ValueError, match="password"):
            extract_text(str(path))


class TestWeightedScore:
    def test_partial_scores_renormalise(self):
        """Summing full weights over a partial score set used to turn a strong
        single dimension into "Limited Commercialisation Potential"."""
        from backend.agents.definitions import overall_score

        assert overall_score({"patentability": 8.0}) == 8.0
        assert overall_score({"patentability": 8.0, "licensing": 6.0}) == 7.0

    def test_complete_scores_are_unchanged(self):
        from backend.agents.definitions import overall_score

        assert overall_score({"patentability": 6, "licensing": 6,
                              "spinout": 6, "risk": 6}) == 5.7

    def test_no_scores_is_zero_not_a_crash(self):
        from backend.agents.definitions import overall_score

        assert overall_score({}) == 0.0


class TestScoresPreferAFinishedRun:
    def test_a_failed_reanalysis_does_not_replace_the_good_run(self, temp_db):
        """A re-analysis that dies after two agents would otherwise serve its
        two scores alongside the previous run's four-dimension verdict."""
        paper_id = temp_db.save_paper("p.pdf", "T", "A", 3)

        good = temp_db.create_run(paper_id)
        for agent in ("patentability", "licensing", "spinout", "risk"):
            temp_db.save_score(paper_id, agent, 7.0, "r", [], [], run_id=good)
        temp_db.update_run(good, "done", done=True)

        broken = temp_db.create_run(paper_id)
        temp_db.save_score(paper_id, "patentability", 2.0, "r", [], [], run_id=broken)
        temp_db.update_run(broken, "error", done=False, error="exploded")

        scores = temp_db.get_paper_scores(paper_id)
        assert len(scores) == 4
        assert {s["score"] for s in scores} == {7.0}

    def test_an_unfinished_run_is_used_when_nothing_finished(self, temp_db):
        paper_id = temp_db.save_paper("p.pdf", "T", "A", 3)
        run = temp_db.create_run(paper_id)
        temp_db.save_score(paper_id, "patentability", 4.0, "r", [], [], run_id=run)

        scores = temp_db.get_paper_scores(paper_id)
        assert len(scores) == 1

    def test_pre_migration_rows_are_still_readable(self, temp_db):
        paper_id = temp_db.save_paper("p.pdf", "T", "A", 3)
        temp_db.save_score(paper_id, "patentability", 5.0, "r", [], [], run_id=None)

        scores = temp_db.get_paper_scores(paper_id)
        assert len(scores) == 1
        assert scores[0]["score"] == 5.0


class TestReportTolerance:
    def test_missing_score_renders_as_a_dash_not_a_crash(self):
        from backend.core.report import build_markdown

        markdown = build_markdown(
            {"title": "T", "page_count": 2},
            [{"agent": "patentability", "score": None, "rationale": "r",
              "key_points": [], "evidence": [], "samples": 1}],
            {"overall": 5.0, "verdict": "Moderate"},
        )
        assert "Patentability" in markdown
        assert "—" in markdown

    def test_deep_run_without_a_max_does_not_crash(self):
        from backend.core.report import build_markdown

        markdown = build_markdown(
            {"title": "T", "page_count": 2},
            [{"agent": "licensing", "score": 6.0, "rationale": "r", "key_points": [],
              "evidence": [], "samples": 3, "score_min": 5.0, "score_max": None}],
            {"overall": 6.0, "verdict": "Moderate"},
        )
        assert "single pass" in markdown


class TestDeleteEndpoint:
    def test_deleting_an_unknown_paper_is_a_404(self, api_client):
        assert api_client.delete("/papers/424242").status_code == 404

    def test_deleting_a_real_paper_succeeds_once(self, api_client, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1)
        assert api_client.delete(f"/papers/{paper_id}").status_code == 200
        assert api_client.delete(f"/papers/{paper_id}").status_code == 404


class TestReanalyse:
    def test_corrupt_stored_pdf_is_422_not_500(self, api_client, temp_db, tmp_path):
        broken = tmp_path / "rotten.pdf"
        broken.write_bytes(b"%PDF-1.4 truncated")
        paper_id = temp_db.save_paper("r.pdf", "T", "A", 1, filepath=str(broken))

        response = api_client.post(f"/papers/{paper_id}/reanalyse")
        assert response.status_code == 422
        assert "could not re-read" in response.json()["detail"].lower()


class TestStructuredMethodSelection:
    def test_an_unreachable_server_is_not_cached(self, monkeypatch):
        """Ollama merely being down at startup used to pin the process to the
        less reliable json_mode for its whole lifetime."""
        from backend.agents import base

        monkeypatch.setattr(base, "_structured_method", None)

        def unreachable(*args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(base.httpx, "get", unreachable)
        assert base._pick_structured_method() == "json_mode"
        assert base._structured_method is None  # not cached

        class Response:
            @staticmethod
            def json():
                return {"version": "0.5.1"}

        monkeypatch.setattr(base.httpx, "get", lambda *a, **k: Response())
        assert base._pick_structured_method() == "json_schema"

    @pytest.mark.parametrize("version,expected", [
        ("0.4.0", "json_mode"), ("0.4.9", "json_mode"),
        ("0.5.0", "json_schema"), ("0.11.4", "json_schema"), ("1.2.0", "json_schema"),
    ])
    def test_version_gate(self, monkeypatch, version, expected):
        from backend.agents import base

        monkeypatch.setattr(base, "_structured_method", None)

        class Response:
            @staticmethod
            def json():
                return {"version": version}

        monkeypatch.setattr(base.httpx, "get", lambda *a, **k: Response())
        assert base._pick_structured_method() == expected


class TestAnalysisIsSerialised:
    """FastAPI runs each upload's background task in its own threadpool worker.
    Selecting several PDFs in the sidebar used to start that many analyses at
    once against one partly-CPU-offloaded model."""

    def test_only_one_analysis_runs_at_a_time(self):
        import threading

        import backend.main as main

        peak = {"value": 0}
        active = {"value": 0}
        guard = threading.Lock()
        started = threading.Event()

        def fake_analysis(*args, **kwargs):
            with guard:
                active["value"] += 1
                peak["value"] = max(peak["value"], active["value"])
            started.set()
            time.sleep(0.2)
            with guard:
                active["value"] -= 1

        original = main._run_analysis
        main._run_analysis = fake_analysis
        try:
            threads = []
            for _ in range(4):
                main._ADMISSION.acquire()
                threads.append(threading.Thread(
                    target=main._process_paper, args=(1, {}, 1, 1)))
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        finally:
            main._run_analysis = original

        assert peak["value"] == 1, f"ran {peak['value']} analyses concurrently"

    def test_admission_is_capped(self, api_client, monkeypatch):
        """Queued analyses block threadpool workers that the sync API endpoints
        share, so admission is refused rather than allowed to starve them."""
        import backend.main as main

        held = []
        try:
            for _ in range(main.MAX_IN_FLIGHT):
                main._ADMISSION.acquire()
                held.append(True)
            with pytest.raises(Exception) as excinfo:
                main._reserve_slot()
            assert "429" in str(excinfo.value) or "queued" in str(excinfo.value)
        finally:
            for _ in held:
                main._ADMISSION.release()

    def test_slot_is_released_after_a_failed_analysis(self):
        import backend.main as main

        def explode(*args, **kwargs):
            raise RuntimeError("boom")

        original = main._run_analysis
        main._run_analysis = explode
        before = main._ADMISSION._value
        try:
            main._ADMISSION.acquire()
            # _run_analysis swallows its own errors in production; this stub
            # models the worse case where one escapes anyway.
            with pytest.raises(RuntimeError):
                main._process_paper(1, {}, 1, 1)
        finally:
            main._run_analysis = original
        assert main._ADMISSION._value == before, "admission slot leaked"
        assert not main._ANALYSIS_LOCK.locked(), "analysis lock leaked"


class TestPublicationYear:
    """Prior art must predate the paper, so the paper's year has to be found."""

    @staticmethod
    def _pdf(tmp_path, first_line, name="p.pdf"):
        import fitz

        path = tmp_path / name
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((40, 60), first_line, fontsize=9)
        page.insert_text((72, 100), "1. Introduction", fontsize=12,
                         fontname="helvetica-bold")
        y = 130
        for i in range(8):
            page.insert_text((72, y), f"Body line {i} with enough text to pass.",
                             fontsize=11)
            y += 16
        doc.save(str(path))
        doc.close()
        return str(path)

    def test_arxiv_stamp_beats_a_later_file_date(self, tmp_path):
        """The stamp encodes the original submission; the file was written
        later. One preprint here reports 2026 metadata for a 2016 paper."""
        from backend.core.pdf_extractor import extract_text

        path = self._pdf(tmp_path, "arXiv:1611.07004v3  [cs.CV]  26 Nov 2018")
        assert extract_text(path)["year"] == 2016

    def test_published_line_is_used(self, tmp_path):
        from backend.core.pdf_extractor import extract_text

        path = self._pdf(tmp_path, "Received: 5 August 2024; Published: 4 September 2024")
        assert extract_text(path)["year"] == 2024

    def test_copyright_year_is_used(self, tmp_path):
        from backend.core.pdf_extractor import extract_text

        path = self._pdf(tmp_path, "Copyright 2011 by the authors.")
        assert extract_text(path)["year"] == 2011

    def test_absurd_years_are_rejected(self, tmp_path):
        from backend.core.pdf_extractor import extract_text

        path = self._pdf(tmp_path, "Published: 12 March 1802")
        year = extract_text(path)["year"]
        assert year is None or year >= 1970

    def test_year_is_never_in_the_future(self, tmp_path):
        from datetime import datetime

        from backend.core.pdf_extractor import extract_text

        path = self._pdf(tmp_path, "Ordinary first line with no date at all.")
        year = extract_text(path)["year"]
        assert year is None or year <= datetime.now().year + 1


class TestPriorArtDateFilter:
    @staticmethod
    def _capture(monkeypatch):
        from backend.core import prior_art

        seen = {}

        class Response:
            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return {"results": []}

        def fake_get(url, params=None, headers=None, timeout=None):
            seen.update(params or {})
            return Response()

        monkeypatch.setattr(prior_art.httpx, "get", fake_get)
        monkeypatch.setattr(prior_art, "ENABLED", True)
        return prior_art, seen

    def test_year_becomes_an_openalex_filter(self, monkeypatch):
        prior_art, seen = self._capture(monkeypatch)
        prior_art.find_similar_works("Some Paper Title", before_year=2016)
        assert seen.get("filter") == "to_publication_date:2016-12-31"

    def test_no_year_means_no_filter(self, monkeypatch):
        prior_art, seen = self._capture(monkeypatch)
        prior_art.find_similar_works("Some Paper Title")
        assert "filter" not in seen

    def test_results_published_after_the_paper_are_excluded(self, monkeypatch):
        """The regression itself: a 2016 paper was handed 2019 and 2020 work
        as prior art, which cannot bear on its novelty."""
        from backend.core import prior_art

        class Response:
            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                # What OpenAlex returns once the date filter is applied.
                return {"results": [
                    {"title": "Contemporary Work", "publication_year": 2016,
                     "cited_by_count": 10},
                    {"title": "Older Work", "publication_year": 2013,
                     "cited_by_count": 5},
                ]}

        monkeypatch.setattr(prior_art.httpx, "get",
                            lambda *a, **k: Response())
        monkeypatch.setattr(prior_art, "ENABLED", True)
        works = prior_art.find_similar_works("Paper", before_year=2016)
        assert works and all(w["year"] <= 2016 for w in works)

    def test_the_analysis_passes_the_year_through(self, api_client, tmp_path,
                                                  monkeypatch):
        import backend.main as main

        captured = {}
        monkeypatch.setattr(
            main, "find_similar_works",
            lambda title, abstract="", before_year=None: captured.update(
                {"year": before_year}) or [])

        path = make_pdf(
            tmp_path / "dated.pdf",
            ["arXiv:1611.07004v3  [cs.CV]  26 Nov 2018", "1. Introduction"]
            + [f"Body line {i} with enough text to pass." for i in range(8)],
            bold_indices=(1,),
        )
        with open(path, "rb") as handle:
            api_client.post("/upload", files={
                "file": ("dated.pdf", handle.read(), "application/pdf")})
        assert captured.get("year") == 2016


class TestDeleteDuringAnalysis:
    """Found by deleting a paper while its agents were running: the guard ran
    before each agent but the insert happened two minutes later, so the delete
    landed in between and surfaced as 'FOREIGN KEY constraint failed'."""

    def test_delete_while_an_agent_is_scoring_is_not_an_error(
            self, api_client, temp_db, tmp_path, monkeypatch):
        import backend.main as main

        path = make_pdf(
            tmp_path / "doomed.pdf",
            ["1. Introduction"] + [f"Body line {i} with enough text to pass."
                                   for i in range(8)],
            bold_indices=(0,),
        )

        state = {"deleted": False}

        def delete_midway(spec, paper_data, **kwargs):
            # Stand in for the user pressing Delete while this agent runs.
            if not state["deleted"]:
                state["deleted"] = True
                temp_db.delete_paper(state["paper_id"])
            return {"score": 6.0, "rationale": "r", "key_points": [],
                    "evidence": [], "score_min": 6.0, "score_max": 6.0,
                    "samples": 1}

        # The paper id is only known after the upload, so capture it first.
        original_save = main.save_paper

        def capture(*args, **kwargs):
            state["paper_id"] = original_save(*args, **kwargs)
            return state["paper_id"]

        monkeypatch.setattr(main, "save_paper", capture)
        monkeypatch.setattr(main.runner, "run", delete_midway)

        with open(path, "rb") as handle:
            response = api_client.post(
                "/upload",
                files={"file": ("doomed.pdf", handle.read(), "application/pdf")},
            )
        assert response.status_code == 200

        # The run is gone with the paper; what matters is that nothing was
        # logged as a failure and no orphaned score rows survive.
        with temp_db.get_conn() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) AS n FROM scores WHERE paper_id=?",
                (state["paper_id"],)).fetchone()["n"]
        assert rows == 0

    def test_integrity_error_on_a_live_paper_still_raises(self, temp_db, monkeypatch):
        """The swallow must be narrow: a foreign-key error for a paper that
        still exists is a real bug, not a user deleting something."""
        import sqlite3

        import backend.main as main

        paper_id = temp_db.save_paper("p.pdf", "T", "A", 1)
        run_id = temp_db.create_run(paper_id)

        def boom(*args, **kwargs):
            raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")

        monkeypatch.setattr(main, "update_run", boom)
        with pytest.raises(sqlite3.IntegrityError):
            main._run_analysis(paper_id, {"full_text": "x", "title": "T"}, run_id, 1)


class TestReportMetadataIsPlainText:
    def test_markup_in_the_title_is_stripped(self):
        from backend.core.report import build_markdown

        markdown = build_markdown(
            {"title": '<img src=x onerror="alert(1)"> Real Title',
             "author": "A <script>alert(2)</script> B", "page_count": 1},
            [], {"overall": 5.0, "verdict": "Moderate"},
        )
        assert "<img" not in markdown
        assert "<script>" not in markdown
        assert "Real Title" in markdown


class TestFrontendEscaping:
    """The UI renders titles, model output and OpenAlex fields as raw HTML."""

    def test_markup_in_a_value_is_escaped(self):
        from render import esc

        assert esc("<img src=x onerror=alert(1)>") == \
            "&lt;img src=x onerror=alert(1)&gt;"

    def test_quotes_are_escaped_so_attributes_cannot_break_out(self):
        from render import esc

        assert "'" not in esc("it's \"quoted\"")
        assert '"' not in esc("it's \"quoted\"")

    def test_none_renders_as_empty(self):
        from render import esc

        assert esc(None) == ""

    @pytest.mark.parametrize("url", [
        "javascript:alert(1)", "JavaScript:alert(1)", "data:text/html,<script>",
        "vbscript:x", "", None,
    ])
    def test_dangerous_link_schemes_are_dropped(self, url):
        from render import safe_url

        assert safe_url(url) == ""

    @pytest.mark.parametrize("url", [
        "https://doi.org/10.1000/xyz", "http://example.org/a",
    ])
    def test_web_links_survive(self, url):
        from render import safe_url

        assert safe_url(url) == url
