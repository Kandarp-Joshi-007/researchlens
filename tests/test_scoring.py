"""Scoring, schema and agent-runner tests."""

import pytest
from pydantic import ValidationError

from backend.agents import runner
from backend.agents.base import AgentVerdict, score_with_agent
from backend.agents.definitions import AGENTS, overall_score, verdict_for


class TestVerdictSchema:
    @pytest.mark.parametrize("score", [-0.1, 10.1, 15, -3])
    def test_rejects_out_of_range_scores(self, score):
        with pytest.raises(ValidationError):
            AgentVerdict(score=score, rationale="x")

    def test_requires_rationale(self):
        with pytest.raises(ValidationError):
            AgentVerdict(score=5)

    def test_lists_default_empty(self):
        verdict = AgentVerdict(score=7, rationale="ok")
        assert verdict.key_points == [] and verdict.evidence == []


class TestScoringMaths:
    def test_weights_sum_to_one(self):
        assert sum(spec["weight"] for spec in AGENTS) == pytest.approx(1.0)

    def test_best_and_worst_cases(self):
        best = {"patentability": 10, "licensing": 10, "spinout": 10, "risk": 0}
        worst = {"patentability": 0, "licensing": 0, "spinout": 0, "risk": 10}
        assert overall_score(best) == 10.0
        assert overall_score(worst) == 0.0

    def test_risk_is_inverted(self):
        low = {"patentability": 5, "licensing": 5, "spinout": 5, "risk": 1}
        high = {"patentability": 5, "licensing": 5, "spinout": 5, "risk": 9}
        assert overall_score(low) > overall_score(high)

    @pytest.mark.parametrize("value,expected", [
        (9.0, "Strong"), (7.5, "Strong"), (7.49, "Moderate"),
        (5.0, "Moderate"), (4.99, "Limited"), (0.0, "Limited"),
    ])
    def test_verdict_thresholds(self, value, expected):
        assert verdict_for(value).startswith(expected)


class TestStructuredOutputFailure:
    def test_raises_rather_than_defaulting(self):
        """A broken agent must surface, not silently score 5.0 as before."""
        class Boom:
            def __or__(self, other):
                return self

            def invoke(self, variables):
                raise ValueError("bad json")

        with pytest.raises(RuntimeError):
            score_with_agent(Boom(), {}, attempts=2)


class TestSampling:
    @staticmethod
    def _stub(monkeypatch, values):
        queue = list(values)

        def fake(prompt, variables, temperature=0.0, attempts=3):
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return AgentVerdict(score=item, rationale=f"r{item}",
                                key_points=["k"], evidence=["e"])

        monkeypatch.setattr(runner, "score_with_agent", fake)

    spec = {"name": "t", "prompt": None, "queries": []}
    paper = {"title": "T", "sections": {"abstract": "A" * 200}, "full_text": "F" * 200}

    def test_reports_median_and_spread(self, monkeypatch):
        self._stub(monkeypatch, [4.0, 7.0, 6.0])
        result = runner.run(self.spec, self.paper, context="ctx", samples=3)
        assert result["score"] == 6.0
        assert (result["score_min"], result["score_max"]) == (4.0, 7.0)
        assert result["rationale"] == "r6.0"

    def test_single_sample_is_exact(self, monkeypatch):
        self._stub(monkeypatch, [8.0])
        result = runner.run(self.spec, self.paper, context="ctx", samples=1)
        assert result["score"] == result["score_min"] == result["score_max"] == 8.0

    def test_survives_one_failed_sample(self, monkeypatch):
        self._stub(monkeypatch, [5.0, ValueError("boom"), 9.0])
        result = runner.run(self.spec, self.paper, context="ctx", samples=3)
        assert result["samples"] == 2

    def test_raises_when_all_samples_fail(self, monkeypatch):
        self._stub(monkeypatch, [ValueError("a"), ValueError("b")])
        with pytest.raises(ValueError):
            runner.run(self.spec, self.paper, context="ctx", samples=2)


class TestContextBuilding:
    def test_falls_back_to_sections_without_embeddings(self):
        paper = {"title": "T", "full_text": "body " * 100,
                 "sections": {"abstract": "ABS " * 50, "methods": "MET " * 50}}
        context = runner.build_context(paper, {"name": "x", "queries": []})
        assert "ABS" in context and "MET" in context

    def test_respects_budget(self):
        paper = {"title": "T", "full_text": "x" * 500_000,
                 "sections": {"abstract": "a" * 500_000}}
        context = runner.build_context(paper, {"name": "x", "queries": []},
                                       budget=5_000)
        assert len(context) <= 5_200

    def test_retrieval_path_respects_budget(self, monkeypatch):
        monkeypatch.setattr(runner, "retrieve_for_queries",
                            lambda queries, paper_id, **kw: ["c" * 20_000] * 5)
        paper = {"title": "T", "full_text": "x" * 1000,
                 "sections": {"abstract": "a" * 50_000}}
        context = runner.build_context(paper, {"name": "x", "queries": ["q"]},
                                       paper_id=1, budget=4_000)
        assert len(context) <= 4_400, "abstract must not overshoot the budget"

    def test_falls_back_when_retrieval_raises(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("chroma down")

        monkeypatch.setattr(runner, "retrieve_for_queries", boom)
        paper = {"title": "T", "full_text": "body",
                 "sections": {"abstract": "ABS " * 20}}
        context = runner.build_context(paper, {"name": "x", "queries": ["q"]},
                                       paper_id=1)
        assert "ABS" in context

    def test_prompts_render_for_every_agent(self):
        for spec in AGENTS:
            variables = {"title": "T", "context": "C"}
            if spec.get("uses_prior_art"):
                variables["prior_art"] = "none"
            assert spec["prompt"].format_messages(**variables)
