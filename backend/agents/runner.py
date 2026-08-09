"""Runs an agent spec against a paper and returns a validated verdict."""

import logging

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
                snippet = abstract[:3000]
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
        context: str = None) -> dict:
    """Score one paper with one agent."""
    if context is None:
        context = build_context(paper_data, spec, paper_id=paper_id)
    verdict: AgentVerdict = score_with_agent(
        spec["prompt"],
        {"title": paper_data.get("title", "Unknown"), "context": context},
    )
    log.info("%s scored %.1f (%d chars of context)",
             spec["name"], verdict.score, len(context))
    return {
        "score": verdict.score,
        "rationale": verdict.rationale,
        "key_points": verdict.key_points[:5],
        "evidence": verdict.evidence[:5],
    }
