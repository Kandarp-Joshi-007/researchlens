"""Extraction tests.

The original extractor put 89% of the sample paper into the 'abstract' bucket
and never found methods, results, discussion or conclusion, so the agents were
reading publisher boilerplate. These tests pin that behaviour down.
"""

import pytest

from backend.core.pdf_extractor import chunk_text, extract_text
from conftest import make_pdf


class TestSectionDetection:
    def test_finds_all_major_sections(self, sample_pdf):
        result = extract_text(sample_pdf)
        headings = {s["heading"].lower() for s in result["sections_ordered"]}
        assert "abstract" in headings
        assert "introduction" in headings
        assert "conclusions" in headings
        # Every canonical bucket the agents fall back on must be populated.
        for bucket in ["abstract", "introduction", "methods", "results",
                       "discussion", "conclusion"]:
            assert result["sections"].get(bucket), f"{bucket} is empty"

    def test_no_single_section_swallows_the_paper(self, sample_pdf):
        result = extract_text(sample_pdf)
        for name, text in result["sections"].items():
            share = len(text) / result["char_count"]
            assert share < 0.75, f"{name} holds {share:.0%} of the paper"

    def test_metadata(self, sample_pdf):
        result = extract_text(sample_pdf)
        assert "Synthetic Data" in result["title"]
        assert result["page_count"] == 38
        assert result["char_count"] > 50_000


class TestBoilerplateRemoval:
    @pytest.mark.parametrize("phrase", [
        "Creative Commons", "Licensee MDPI", "Academic Editor",
        "Citation: Goyal", "doi.org", "Received: 28 July", "Copyright:",
    ])
    def test_publisher_boilerplate_is_stripped(self, sample_pdf, phrase):
        assert phrase.lower() not in extract_text(sample_pdf)["full_text"].lower()

    def test_references_are_dropped(self, sample_pdf):
        result = extract_text(sample_pdf)
        assert "references" not in result["sections"]

    def test_running_headers_are_dropped(self, sample_pdf):
        # "Electronics 2024, 13, 3509  N of 38" repeats on every page.
        assert "of 38" not in extract_text(sample_pdf)["full_text"]


class TestFallbacks:
    def test_pdf_without_headings_still_yields_text(self, tmp_path):
        path = make_pdf(tmp_path / "plain.pdf", [
            "A new catalytic process for hydrogen storage is presented.",
            "The method achieves 40 percent higher yield at lower cost.",
            "Results were validated across twelve independent trials.",
        ])
        result = extract_text(path)
        assert "catalytic process" in result["full_text"]
        assert result["char_count"] > 50

    def test_empty_pdf_raises(self, tmp_path):
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")
        with pytest.raises(Exception):
            extract_text(str(path))

    def test_table_and_figure_captions_are_not_headings(self, tmp_path):
        path = make_pdf(
            tmp_path / "caps.pdf",
            ["1. Introduction", "Body text here about the study.",
             "Table 1. Summary of results.", "More body text follows."],
            bold_indices=(0, 2),
        )
        headings = {s["heading"] for s in extract_text(path)["sections_ordered"]}
        assert not any(h.lower().startswith("table") for h in headings)


class TestChunking:
    def test_overlap_and_sizes(self):
        words = " ".join(f"w{i}" for i in range(2000))
        chunks = chunk_text(words, chunk_size=800, overlap=100)
        assert [len(c.split()) for c in chunks] == [800, 800, 600]
        assert chunks[0].split()[-100:] == chunks[1].split()[:100]

    def test_edge_cases(self):
        assert chunk_text("") == []
        assert len(chunk_text("a b c")) == 1
        # Must terminate rather than loop when overlap >= chunk_size.
        assert len(chunk_text("a b c d e", chunk_size=2, overlap=5)) >= 1
