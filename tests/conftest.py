import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
