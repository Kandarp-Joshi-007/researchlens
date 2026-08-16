import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# The frontend is a script, not a package; its helper modules are imported by
# name so they can be tested without executing the Streamlit app.
sys.path.insert(0, str(ROOT / "frontend"))

SAMPLE_PDF = ROOT / "electronics-13-03509 (1).pdf"


@pytest.fixture(scope="session")
def sample_pdf():
    if not SAMPLE_PDF.exists():
        pytest.skip("sample PDF not present")
    return str(SAMPLE_PDF)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the database module at a throwaway file."""
    from backend.core import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    return database


@pytest.fixture
def api_client(temp_db, monkeypatch, tmp_path):
    """TestClient with the LLM, embeddings and OpenAlex stubbed out.

    Uploads go to a throwaway directory; otherwise the suite litters the real
    uploads/ folder with test PDFs.
    """
    from fastapi.testclient import TestClient

    import backend.main as main

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(main, "UPLOAD_DIR", uploads)

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


def make_pdf(path, lines, bold_indices=()):
    """Build a small PDF; bold lines stand in for section headings."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for index, line in enumerate(lines):
        font = "helvetica-bold" if index in bold_indices else "helvetica"
        page.insert_text((72, y), line, fontsize=11, fontname=font)
        y += 18
    doc.save(str(path))
    doc.close()
    return str(path)
