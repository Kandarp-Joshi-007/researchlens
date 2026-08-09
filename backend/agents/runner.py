"""Runs an agent spec against a paper and returns a validated verdict."""

import logging

from .base import AgentVerdict, context_budget_chars, score_with_agent

log = logging.getLogger(__name__)

# Sections used when retrieval is unavailable, in order of usefulness.
_FALLBACK_SECTIONS = ["abstract", "introduction", "methods", "results",
                      "discussion", "conclusion"]


def build_context(paper_data: dict, spec: dict, budget: int = None) -> str:
    """Assemble the paper excerpts an agent sees, within the context budget."""
    budget = budget or context_budget_chars()
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


def run(spec: dict, paper_data: dict, context: str = None) -> dict:
    """Score one paper with one agent."""
    context = context if context is not None else build_context(paper_data, spec)
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
