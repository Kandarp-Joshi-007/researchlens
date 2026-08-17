"""Prior-art lookup via OpenAlex.

Novelty judged by a language model from the paper alone is guesswork: the model
has no way to know what else exists. Retrieving genuinely similar published work
turns the patentability assessment into a comparison against real evidence.

OpenAlex is free and needs no key. Supplying a contact email joins the polite
pool and gets faster, more reliable service.
"""

import logging
import os
import re
from typing import List, Optional

import httpx

log = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org/works"
CONTACT_EMAIL = os.getenv("RESEARCHLENS_CONTACT_EMAIL", "")
TIMEOUT = float(os.getenv("RESEARCHLENS_HTTP_TIMEOUT", "12"))

# Prior-art lookup is a nice-to-have; never let it block or fail an analysis.
ENABLED = os.getenv("RESEARCHLENS_PRIOR_ART", "1") not in ("0", "false", "False")


def _clean_query(text: str) -> str:
    """OpenAlex search rejects boolean operators and punctuation-heavy strings."""
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\b(AND|OR|NOT)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:250]


def find_similar_works(title: str, abstract: str = "", limit: int = 8,
                       before_year: Optional[int] = None) -> List[dict]:
    """Published work similar to this paper, most cited first.

    `before_year` restricts results to work published no later than that year.
    Without it the search returns whatever is most relevant, which for a 2016
    paper included 2019 and 2020 work — papers that cite it rather than precede
    it. Nothing published after a paper can bear on its novelty, so feeding
    those to the patentability agent as "prior art" is simply wrong.

    The bound is inclusive because only the year is known: same-year work may
    well have come first, and wrongly hiding real prior art is the worse error.

    Returns an empty list on any failure — offline use stays fully functional.
    """
    if not ENABLED or not title:
        return []

    query = _clean_query(title)
    if not query:
        return []

    params = {
        "search": query,
        "per-page": limit,
        "select": "id,title,publication_year,cited_by_count,doi,authorships,primary_location",
        "sort": "relevance_score:desc",
    }
    if before_year:
        params["filter"] = f"to_publication_date:{int(before_year)}-12-31"
    headers = {"User-Agent": f"ResearchLens/1.0 ({CONTACT_EMAIL or 'local install'})"}
    if CONTACT_EMAIL:
        params["mailto"] = CONTACT_EMAIL

    try:
        response = httpx.get(OPENALEX_URL, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as exc:
        log.warning("OpenAlex lookup failed (%s); continuing without prior art", exc)
        return []

    works = []
    for item in results:
        item_title = (item.get("title") or "").strip()
        if not item_title:
            continue
        # Drop the paper itself from its own prior-art list.
        if _normalise(item_title) == _normalise(title):
            continue
        works.append({
            "title": item_title,
            "year": item.get("publication_year"),
            "citations": item.get("cited_by_count", 0),
            "doi": item.get("doi"),
            "venue": _venue(item),
            "authors": _first_authors(item),
        })
    return works


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _venue(item: dict) -> Optional[str]:
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    return source.get("display_name")


def _first_authors(item: dict, limit: int = 3) -> str:
    names = []
    for authorship in (item.get("authorships") or [])[:limit]:
        author = authorship.get("author") or {}
        if author.get("display_name"):
            names.append(author["display_name"])
    suffix = " et al." if len(item.get("authorships") or []) > limit else ""
    return ", ".join(names) + suffix


def format_for_prompt(works: List[dict], limit: int = 8) -> str:
    """Render prior art as compact lines for inclusion in an agent prompt."""
    if not works:
        return "No prior-art records were retrieved (offline or no matches)."
    lines = []
    for work in works[:limit]:
        year = work.get("year") or "n.d."
        venue = f", {work['venue']}" if work.get("venue") else ""
        lines.append(
            f"- \"{work['title']}\" ({year}{venue}) — cited {work.get('citations', 0)} times"
        )
    return "\n".join(lines)
