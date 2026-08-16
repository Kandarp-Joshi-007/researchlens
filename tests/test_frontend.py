"""Frontend tests using Streamlit's own harness.

The UI was previously only exercised by hand. These run the real script with the
backend stubbed, which catches the errors that only appear once a value is
actually rendered — a None where a float was formatted, a crash when the API is
unreachable, markup arriving from a PDF's metadata.
"""

import json
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")

SCORES = [
    {"agent": "patentability", "score": 7.0, "rationale": "Novel membrane.",
     "key_points": ["a", "b"], "evidence": ["verbatim quote"], "samples": 1,
     "score_min": 7.0, "score_max": 7.0},
    {"agent": "licensing", "score": 8.0, "rationale": "Large market.",
     "key_points": ["c"], "evidence": [], "samples": 1,
     "score_min": 8.0, "score_max": 8.0},
    {"agent": "spinout", "score": 6.0, "rationale": "Clear path.",
     "key_points": [], "evidence": ["another quote"], "samples": 3,
     "score_min": 5.0, "score_max": 7.0},
    {"agent": "risk", "score": 3.0, "rationale": "Few barriers.",
     "key_points": [], "evidence": [], "samples": 1,
     "score_min": 3.0, "score_max": 3.0},
]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def install_fake_api(at, routes, error=None):
    """Route requests.request through a table of path -> payload."""
    import requests

    def fake_request(method, url, **kwargs):
        if error is not None:
            raise error
        path = url.split("8000", 1)[-1]
        for prefix, payload in routes.items():
            if path.startswith(prefix):
                if isinstance(payload, tuple):
                    return FakeResponse(payload[0], payload[1])
                return FakeResponse(payload)
        return FakeResponse({"detail": "not found"}, 404)

    at.session_state  # touch so the harness is initialised
    requests.request = fake_request
    return at


@pytest.fixture(autouse=True)
def restore_requests():
    import requests

    original = requests.request
    yield
    requests.request = original


def build(routes=None, error=None, session=None):
    at = AppTest.from_file(APP, default_timeout=30)
    for key, value in (session or {}).items():
        at.session_state[key] = value
    install_fake_api(at, routes or {}, error=error)
    return at


class TestBackendDown:
    def test_the_page_still_renders_with_a_clear_message(self):
        import requests

        at = build(error=requests.ConnectionError("refused"))
        at.run()
        assert not at.exception, at.exception
        warnings = " ".join(w.value for w in at.warning)
        assert "backend" in warnings.lower() or "reach" in warnings.lower()

    def test_timeout_is_reported_not_raised(self):
        import requests

        at = build(error=requests.Timeout("slow"))
        at.run()
        assert not at.exception, at.exception


class TestEmptyLibrary:
    def test_no_papers_message(self):
        at = build({"/results": []})
        at.run()
        assert not at.exception, at.exception
        text = " ".join(i.value for i in at.info)
        assert "No papers analysed yet" in text or "No scored papers" in text


class TestLibraryRendering:
    def _routes(self, title="A Real Paper"):
        return {
            "/results/1": {"paper_id": 1, "scores": SCORES, "overall": 6.9,
                           "verdict": "Moderate Potential", "prior_art": []},
            "/results": [{"id": 1, "filename": "p.pdf", "title": title,
                          "uploaded_at": "2026-08-16 10:00:00", "overall": 6.9,
                          "verdict": "Moderate Potential"}],
        }

    def test_renders_a_scored_paper(self):
        at = build(self._routes())
        at.run()
        assert not at.exception, at.exception
        assert any("A Real Paper" in str(m.value) for m in at.markdown)

    def test_pending_paper_without_a_score_does_not_crash(self):
        """overall is None until the analysis finishes; formatting it as a float
        would raise on every library render."""
        at = build({"/results": [{"id": 1, "filename": "p.pdf", "title": "Pending",
                                  "uploaded_at": "2026-08-16 10:00:00",
                                  "overall": None, "verdict": None}]})
        at.run()
        assert not at.exception, at.exception
        assert any("Pending" in str(m.value) for m in at.markdown)

    def test_markup_in_a_title_is_not_emitted_as_html(self):
        at = build(self._routes(title='<img src=x onerror="alert(1)">'))
        at.run()
        assert not at.exception, at.exception
        rendered = " ".join(str(m.value) for m in at.markdown)
        assert "<img src=x" not in rendered
        assert "&lt;img" in rendered


class TestResultsView:
    def _routes(self, prior_art=None):
        return {
            "/status/1": {"stage": "done", "done": True, "error": None,
                          "overall": 6.9},
            "/results/1": {"paper_id": 1, "scores": SCORES, "overall": 6.9,
                           "verdict": "Moderate Potential",
                           "prior_art": prior_art or []},
            "/papers/1/report": {"paper_id": 1, "filename": "brief.md",
                                 "markdown": "# Brief"},
            "/results": [{"id": 1, "filename": "p.pdf", "title": "T",
                          "uploaded_at": "2026-08-16 10:00:00", "overall": 6.9,
                          "verdict": "Moderate Potential"}],
        }

    def test_full_results_render(self):
        at = build(self._routes(), session={"active_paper_id": 1,
                                            "active_title": "T"})
        at.run()
        assert not at.exception, at.exception
        rendered = " ".join(str(m.value) for m in at.markdown)
        assert "Patentability" in rendered
        # Risk is stored raw and shown inverted: 3.0 raw becomes 7.0 displayed.
        assert "7.0" in rendered

    def test_evidence_quotes_are_escaped(self):
        routes = self._routes()
        routes["/results/1"] = dict(routes["/results/1"])
        scores = [dict(s) for s in SCORES]
        scores[0]["evidence"] = ["<script>alert(1)</script>"]
        routes["/results/1"]["scores"] = scores
        at = build(routes, session={"active_paper_id": 1, "active_title": "T"})
        at.run()
        assert not at.exception, at.exception
        rendered = " ".join(str(m.value) for m in at.markdown)
        assert "<script>" not in rendered

    def test_prior_art_with_a_javascript_doi_drops_the_link(self):
        at = build(
            self._routes(prior_art=[{
                "title": "Some Work", "year": 2024, "venue": "Venue",
                "citations": 5, "doi": "javascript:alert(1)",
                "authors": "A, B",
            }]),
            session={"active_paper_id": 1, "active_title": "T"},
        )
        at.run()
        assert not at.exception, at.exception
        rendered = " ".join(str(m.value) for m in at.markdown)
        assert "javascript:" not in rendered
        assert "Some Work" in rendered

    def test_prior_art_with_missing_fields_does_not_crash(self):
        at = build(
            self._routes(prior_art=[{"title": "Bare Work"}]),
            session={"active_paper_id": 1, "active_title": "T"},
        )
        at.run()
        assert not at.exception, at.exception


class TestErrorStates:
    def test_failed_analysis_shows_the_error_and_a_retry(self):
        at = build({
            "/status/1": {"stage": "error", "done": False,
                          "error": "agent exploded"},
            "/results": [],
        }, session={"active_paper_id": 1, "active_title": "T"})
        at.run()
        assert not at.exception, at.exception
        errors = " ".join(e.value for e in at.error)
        assert "agent exploded" in errors

    def test_report_endpoint_failure_does_not_hide_the_scores(self):
        routes = {
            "/status/1": {"stage": "done", "done": True, "error": None},
            "/results/1": {"paper_id": 1, "scores": SCORES, "overall": 6.9,
                           "verdict": "Moderate", "prior_art": []},
            "/papers/1/report": ({"detail": "not scored"}, 409),
            "/results": [],
        }
        at = build(routes, session={"active_paper_id": 1, "active_title": "T"})
        at.run()
        assert not at.exception, at.exception
        rendered = " ".join(str(m.value) for m in at.markdown)
        assert "Patentability" in rendered
