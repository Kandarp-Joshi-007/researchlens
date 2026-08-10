"""Runs an agent spec against a paper and returns a validated verdict."""

import logging
from statistics import median

from ..core.prior_art import format_for_prompt
from ..core.vectorstore import retrieve_for_queries
from .base import AgentVerdict, context_budget_chars, score_with_agent

log = logging.getLogger(__name__)

# Sections used when retrieval is unavailable, in order of usefulness.
_FALLBACK_SECTIONS = ["abstract", "introduction", "methods", "results",
                      "discussion", "conclusion"]


def build_context(paper_data: dict, spec: dict, paper_id: int = None,
                  budget: int = None) -> str:
    """Assemble the paper excerpts an agent sees, within the context budget.

    Prefers retrieval: each agent pulls the passages matching its own concerns
    rather than a fixed slice of the paper. The abstract is always included so
    the agent has the paper's overview alongside targeted evidence.
    """
    budget = budget or context_budget_chars()

    if paper_id is not None:
        try:
            retrieved = retrieve_for_queries(spec["queries"], paper_id)
        except Exception as exc:
            log.warning("Retrieval failed for paper %s (%s); using sections",
                        paper_id, exc)
            retrieved = []

        if retrieved:
            parts = []
            used = 0
            abstract = (paper_data.get("sections") or {}).get("abstract", "").strip()
            if abstract:
                snippet = abstract[:min(3000, budget)]
                parts.append(f"[ABSTRACT]\n{snippet}")
                used += len(snippet)

            for chunk in retrieved:
                room = budget - used
                if room <= 400:
                    break
                parts.append(f"[EXCERPT]\n{chunk[:room]}")
                used += min(len(chunk), room)
            return "\n\n".join(parts)

    return _sections_context(paper_data, budget)


def _sections_context(paper_data: dict, budget: int) -> str:
    """Fallback when no embeddings are available: walk the canonical sections."""
    parts = []
    used = 0

    for name in _FALLBACK_SECTIONS:
        text = (paper_data.get("sections") or {}).get(name, "").strip()
        if not text:
            continue
        room = budget - used
        if room <= 200:
            break
        snippet = text[:room]
        parts.append(f"[{name.upper()}]\n{snippet}")
        used += len(snippet)

    if not parts:
        parts.append((paper_data.get("full_text") or "")[:budget])

    return "\n\n".join(parts)


def run(spec: dict, paper_data: dict, paper_id: int = None,
        context: str = None, prior_art: list = None, samples: int = 1) -> dict:
    """Score one paper with one agent.

    With samples > 1 the agent is asked repeatedly at a non-zero temperature and
    the median is reported alongside the spread. Temperature 0 makes a score
    repeatable but not necessarily reliable; sampling shows how stable the
    judgement actually is. Costs one full inference per sample.
    """
    if context is None:
        context = build_context(paper_data, spec, paper_id=paper_id)

    variables = {"title": paper_data.get("title", "Unknown"), "context": context}
    if spec.get("uses_prior_art"):
        variables["prior_art"] = format_for_prompt(prior_art or [])

    samples = max(1, samples)
    verdicts = []
    for index in range(samples):
        # Deterministic for a single pass; varied when sampling for spread.
        temperature = 0.0 if samples == 1 else 0.7
        try:
            verdicts.append(score_with_agent(spec["prompt"], variables,
                                             temperature=temperature))
        except Exception:
            # Tolerate a failed sample as long as one attempt succeeded.
            if index == samples - 1 and not verdicts:
                raise
            log.warning("%s sample %d failed; continuing", spec["name"], index + 1)

    scores = [v.score for v in verdicts]
    median_score = median(scores)
    # Report the sample nearest the median so prose matches the headline number.
    chosen = min(verdicts, key=lambda v: abs(v.score - median_score))

    log.info("%s scored %.1f (range %.1f-%.1f over %d sample(s), %d chars context)",
             spec["name"], median_score, min(scores), max(scores),
             len(scores), len(context))

    return {
        "score": round(median_score, 1),
        "rationale": chosen.rationale,
        "key_points": chosen.key_points[:5],
        "evidence": chosen.evidence[:5],
        "score_min": min(scores),
        "score_max": max(scores),
        "samples": len(scores),
    }
