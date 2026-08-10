"""Commercialisation brief rendering."""

from backend.core.report import build_markdown

PAPER = {"title": "A Test Paper", "filename": "t.pdf", "author": "A. Author",
         "page_count": 12}
SCORES = [
    {"agent": "patentability", "score": 8.0, "rationale": "Novel method.",
     "key_points": ["new"], "evidence": ["a verbatim quote"], "samples": 1},
    {"agent": "risk", "score": 7.0, "rationale": "Regulatory hurdles.",
     "key_points": ["regulated"], "evidence": [], "samples": 3,
     "score_min": 6.0, "score_max": 9.0},
]
SUMMARY = {"overall": 6.4, "verdict": "Moderate Potential"}


class TestReport:
    def test_includes_headline_and_content(self):
        markdown = build_markdown(PAPER, SCORES, SUMMARY)
        assert "# Commercialisation Brief: A Test Paper" in markdown
        assert "6.4 / 10" in markdown
        assert "Moderate Potential" in markdown
        assert "Novel method." in markdown
        assert "> a verbatim quote" in markdown

    def test_risk_is_shown_the_same_way_up_as_other_dimensions(self):
        markdown = build_markdown(PAPER, SCORES, SUMMARY)
        # Stored risk 7.0 is high risk, presented as 3.0 alongside the rest.
        assert "**3.0**" in markdown
        assert "inverted" in markdown

    def test_reports_confidence_when_sampled(self):
        markdown = build_markdown(PAPER, SCORES, SUMMARY)
        assert "over 3 samples" in markdown
        assert "single pass" in markdown

    def test_prior_art_section_both_ways(self):
        empty = build_markdown(PAPER, SCORES, SUMMARY, [])
        assert "No prior-art records" in empty

        works = [{"title": "Related Work", "year": 2022, "venue": "Journal",
                  "citations": 42, "doi": "https://doi.org/10.1/x"}]
        filled = build_markdown(PAPER, SCORES, SUMMARY, works)
        assert "[Related Work](https://doi.org/10.1/x)" in filled
        assert "| 42 |" in filled

    def test_handles_missing_scores_and_summary(self):
        markdown = build_markdown(PAPER, [], None)
        assert "Not yet scored" in markdown
        assert "| — |" in markdown

    def test_pipes_in_titles_do_not_break_the_table(self):
        works = [{"title": "A | B", "year": 2020, "venue": "V | W", "citations": 1}]
        markdown = build_markdown(PAPER, SCORES, SUMMARY, works)
        assert "A \\| B" in markdown
