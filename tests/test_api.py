"""API tests. The LLM and network are stubbed so the suite stays fast and offline."""

import pytest
from fastapi.testclient import TestClient

from conftest import make_pdf


@pytest.fixture
def client(temp_db, monkeypatch):
    import backend.main as main

    # Never call Ollama or OpenAlex from the test suite.
    monkeypatch.setattr(main, "add_chunks", lambda *a, **k: None)
    monkeypatch.setattr(main, "find_similar_works", lambda *a, **k: [])
    monkeypatch.setattr(
        main.runner, "run",
        lambda spec, paper_data, **kw: {
            "score": 6.0, "rationale": "stub", "key_points": ["k"],
            "evidence": ["quote"], "score_min": 6.0, "score_max": 6.0,
            "samples": 1,
        },
    )
    return TestClient(main.app)


class TestUploadValidation:
    def test_rejects_non_pdf(self, client):
        response = client.post("/upload", files={"file": ("a.txt", b"hi", "text/plain")})
        assert response.status_code == 400

    def test_rejects_empty_pdf_without_a_500(self, client):
        response = client.post("/upload", files={"file": ("a.pdf", b"", "application/pdf")})
        assert response.status_code == 400
        assert "could not read" in response.json()["detail"].lower()

    def test_rejects_pdf_with_no_extractable_text(self, client, tmp_path):
        import fitz

        path = tmp_path / "blank.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(path))
        doc.close()
        response = client.post(
            "/upload",
            files={"file": ("blank.pdf", path.read_bytes(), "application/pdf")},
        )
        assert response.status_code == 422


class TestPipeline:
    @staticmethod
    def _pdf(tmp_path):
        return make_pdf(
            tmp_path / "paper.pdf",
            ["1. Introduction",
             "We present a new catalytic membrane for hydrogen separation.",
             "2. Methods",
             "Membranes were synthesised by chemical vapour deposition and tested.",
             "3. Conclusions",
             "The membrane doubles throughput at a quarter of the cost."],
            bold_indices=(0, 2, 4),
        )

    def test_end_to_end_scores_a_paper(self, client, tmp_path):
        path = self._pdf(tmp_path)
        with open(path, "rb") as handle:
            response = client.post(
                "/upload", files={"file": ("paper.pdf", handle.read(), "application/pdf")}
            )
        assert response.status_code == 200
        paper_id = response.json()["paper_id"]

        # TestClient runs background tasks synchronously on response close.
        status = client.get(f"/status/{paper_id}").json()
        assert status["done"] is True, status
        assert status["error"] is None

        result = client.get(f"/results/{paper_id}").json()
        assert len(result["scores"]) == 4
        # Every stub scores 6.0; risk is inverted, so
        # 6*.30 + 6*.30 + 6*.25 + (10-6)*.15 = 5.7
        assert result["overall"] == 5.7
        assert all(s["evidence"] == ["quote"] for s in result["scores"])

    def test_duplicate_upload_is_served_from_cache(self, client, tmp_path):
        path = self._pdf(tmp_path)
        payload = open(path, "rb").read()
        first = client.post("/upload", files={"file": ("p.pdf", payload, "application/pdf")})
        second = client.post("/upload", files={"file": ("p.pdf", payload, "application/pdf")})
        assert second.json().get("duplicate") is True
        assert second.json()["paper_id"] == first.json()["paper_id"]

    def test_failure_is_recorded_on_the_run(self, client, tmp_path, monkeypatch):
        import backend.main as main

        def explode(*args, **kwargs):
            raise RuntimeError("agent exploded")

        monkeypatch.setattr(main.runner, "run", explode)
        # insert_text does not wrap, so use several short lines rather than
        # one long one that would run off the page and vanish.
        path = make_pdf(
            tmp_path / "boom.pdf",
            ["1. Introduction"] + [f"Body line {i} with enough text to pass the guard."
                                   for i in range(8)],
            bold_indices=(0,),
        )
        with open(path, "rb") as handle:
            response = client.post(
                "/upload", files={"file": ("boom.pdf", handle.read(), "application/pdf")}
            )
        paper_id = response.json()["paper_id"]
        status = client.get(f"/status/{paper_id}").json()
        assert status["stage"] == "error"
        assert "agent exploded" in status["error"]


class TestEndpoints:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_unknown_paper_404s(self, client):
        assert client.get("/results/9999").status_code == 404
        assert client.get("/papers/9999/report").status_code == 404
        assert client.post("/papers/9999/reanalyse").status_code == 404

    def test_status_of_unknown_paper(self, client):
        assert client.get("/status/9999").json()["stage"] == "unknown"

    def test_report_requires_scores(self, client, temp_db):
        paper_id = temp_db.save_paper("f.pdf", "T", "A", 1)
        assert client.get(f"/papers/{paper_id}/report").status_code == 409
