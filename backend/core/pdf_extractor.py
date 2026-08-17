"""PDF text extraction with structure-aware section detection.

Headings in academic PDFs are usually the same font size as body text and
distinguished only by weight, so detection keys off the bold flag combined
with a numbering pattern rather than font size.
"""

import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # pymupdf

# A numbered heading: "3. Research Methodology", "4.2 Ablation Study".
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z].{2,70})$")

# Unnumbered headings that appear in most papers.
_KNOWN_HEADING = re.compile(
    r"^(abstract|introduction|background|related work|methods?|methodology|"
    r"materials and methods|results?|findings?|discussion|conclusions?|"
    r"references|bibliography|acknowledge?ments?|appendix)\b[:.]?\s*$",
    re.I,
)

# Publisher/administrative blocks that carry no analytical value.
_BACK_MATTER = re.compile(
    r"^(references|bibliography|acknowledge?ments?|funding|conflicts? of interest|"
    r"author contributions|data availability|institutional review board|"
    r"informed consent|abbreviations|appendix|supplementary)\b",
    re.I,
)

# Back matter written as a bold run-in rather than its own heading,
# e.g. "Funding: This research received no external funding."
_BACK_MATTER_RUNIN = re.compile(
    r"^(funding|conflicts? of interest|author contributions|data availability|"
    r"acknowledge?ments?|institutional review board|informed consent)\b[^.]{0,40}:",
    re.I,
)

# Lines that are pure publisher boilerplate wherever they appear.
_BOILERPLATE = re.compile(
    r"(creative commons|licensee mdpi|this article is an open access|"
    r"^citation:|^doi:|^https?://|^academic editor|^received:|^revised:|"
    r"^accepted:|^published:|^copyright:|all rights reserved|"
    r"^\(?\s*©|^issn|^www\.)",
    re.I,
)

# Canonical section buckets. First matching keyword wins.
_SECTION_MAP = [
    ("references", ["reference", "bibliography"]),
    ("related_work", ["related work", "literature review", "prior work", "related studies"]),
    ("methods", ["method", "methodology", "approach", "materials", "experimental setup",
                 "study design", "implementation"]),
    ("results", ["result", "finding", "evaluation", "experiment", "performance"]),
    ("discussion", ["discussion", "analysis", "challenge", "limitation", "future research",
                    "future work", "implication"]),
    ("conclusion", ["conclusion", "concluding", "summary"]),
    ("introduction", ["introduction", "background", "motivation", "overview"]),
    ("abstract", ["abstract"]),
]


def _canonical_section(heading: str) -> Optional[str]:
    """Map a heading title to a canonical section name, or None if unrecognised."""
    low = heading.lower()
    for name, keywords in _SECTION_MAP:
        if any(k in low for k in keywords):
            return name
    return None


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _page_blocks(page) -> list:
    """Return blocks of (text, is_bold, size) lines, in reading order.

    Grouping stays at block level so that multi-line publisher boilerplate
    (the MDPI page-1 citation sidebar, for instance) can be dropped whole:
    only its first line carries the give-away marker.
    Handles two-column layouts by emitting the left column before the right.
    """
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("lines")]
    if not blocks:
        return []

    width = page.rect.width
    mid = width / 2

    def column_key(block):
        x0, _, x1, _ = block["bbox"]
        # Full-width blocks (titles, spanning figures) sort with the left column.
        if (x1 - x0) > 0.65 * width:
            return 0
        return 0 if (x0 + x1) / 2 < mid else 1

    # Only apply column ordering when the page really has two columns.
    centers = [((b["bbox"][0] + b["bbox"][2]) / 2) for b in blocks]
    two_col = (
        any(c < mid * 0.9 for c in centers)
        and any(c > mid * 1.1 for c in centers)
        and all((b["bbox"][2] - b["bbox"][0]) < 0.65 * width for b in blocks
                if (b["bbox"][0] + b["bbox"][2]) / 2 > mid * 1.1)
    )
    if two_col:
        blocks.sort(key=lambda b: (column_key(b), b["bbox"][1]))
    else:
        blocks.sort(key=lambda b: b["bbox"][1])

    out = []
    for block in blocks:
        lines = []
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            lines.append((text, _is_bold(spans[0]), spans[0]["size"]))
        if lines:
            out.append({"bbox": block["bbox"], "width": width, "lines": lines})
    return out


def _body_left_edge(pages_blocks: list) -> float:
    """Left edge of the main text column.

    Publishers park citation and copyright notices in the margin beside the
    body column; those blocks sit entirely to the left of this edge.
    """
    xs = Counter()
    for blocks in pages_blocks:
        for block in blocks:
            x0, _, x1, _ = block["bbox"]
            if (x1 - x0) > 0.45 * block["width"]:
                xs[round(x0 / 5) * 5] += 1
    return float(xs.most_common(1)[0][0]) if xs else 0.0


def _is_margin_matter(block: dict, body_left: float) -> bool:
    """True for narrow blocks sitting wholly outside the body column."""
    x0, _, x1, _ = block["bbox"]
    narrow = (x1 - x0) < 0.35 * block["width"]
    return narrow and x1 < body_left - 5


def _running_lines(pages_blocks: list, page_count: int) -> set:
    """Short lines repeated across many pages — running headers and footers."""
    if page_count < 4:
        return set()
    counts = Counter()
    for blocks in pages_blocks:
        for text, _, _ in (line for block in blocks for line in block["lines"]):
            if len(text) < 90:
                # Normalise page numbers so "3 of 38" and "4 of 38" collapse together.
                counts[re.sub(r"\d+", "#", text)] += 1
    threshold = max(3, int(page_count * 0.3))
    return {key for key, n in counts.items() if n >= threshold}


def _looks_like_heading(text: str, is_bold: bool) -> Optional[str]:
    """Return the heading title if this line is a section heading."""
    if len(text) > 90:
        return None
    if text.lower().startswith(("table", "figure", "fig.", "eq.", "algorithm")):
        return None

    match = _NUMBERED_HEADING.match(text)
    if match and is_bold:
        # Reject sentence-like run-ins: real headings rarely end in a period
        # and rarely contain sentence punctuation.
        title = match.group(2).strip()
        if title.endswith(".") or "," in title[:-1] and len(title) > 60:
            return None
        return title

    if _KNOWN_HEADING.match(text):
        return _KNOWN_HEADING.match(text).group(1).strip()

    return None


def extract_text(pdf_path: str) -> dict:
    """Extract structured text and metadata from a PDF.

    The document is closed on every path. PyMuPDF holds the file open, and on
    Windows that lock makes the caller's cleanup unlink() fail, stranding the
    upload on disk whenever extraction raises.
    """
    doc = fitz.open(pdf_path)
    try:
        return _extract(doc, pdf_path)
    finally:
        doc.close()


def _extract(doc, pdf_path: str) -> dict:
    if doc.needs_pass:
        raise ValueError("PDF is password-protected")

    page_count = int(doc.page_count)

    pages_blocks = [_page_blocks(page) for page in doc]
    running = _running_lines(pages_blocks, page_count)

    ordered = []  # [(heading, [paragraphs])] in document order
    current_heading = "front_matter"
    current_body = []
    hit_back_matter = False

    body_left = _body_left_edge(pages_blocks)

    for block in (b for blocks in pages_blocks for b in blocks):
        if _is_margin_matter(block, body_left):
            continue
        # Publisher boilerplate spans several lines but only marks its first,
        # so one hit disqualifies the whole block.
        if any(_BOILERPLATE.search(text) for text, _, _ in block["lines"]):
            continue

        for text, is_bold, _size in block["lines"]:
            if re.sub(r"\d+", "#", text) in running:
                continue

            # Back matter can appear as a bold run-in ("Funding: ...") rather
            # than a standalone heading.
            if _BACK_MATTER_RUNIN.match(text):
                hit_back_matter = True
                continue

            heading = _looks_like_heading(text, is_bold)
            if heading:
                ordered.append((current_heading, current_body))
                current_heading = heading
                current_body = []
                hit_back_matter = bool(_BACK_MATTER.match(heading))
                continue

            if hit_back_matter:
                continue

            # MDPI-style run-in abstract: "Abstract: ..." starts the content.
            abstract_match = re.match(r"^abstract\s*[:.]\s*(.+)", text, re.I)
            if abstract_match and current_heading == "front_matter":
                ordered.append((current_heading, current_body))
                current_heading = "Abstract"
                current_body = [abstract_match.group(1).strip()]
                continue

            current_body.append(text)

    ordered.append((current_heading, current_body))

    # Build canonical sections and an ordered list, dropping front/back matter.
    sections = {}
    sections_ordered = []
    content_parts = []
    for heading, body in ordered:
        body_text = " ".join(body).strip()
        if not body_text or heading == "front_matter":
            continue
        if _BACK_MATTER.match(heading):
            continue

        sections_ordered.append({"heading": heading, "text": body_text})
        content_parts.append(f"## {heading}\n{body_text}")

        canonical = _canonical_section(heading)
        if canonical and canonical != "references":
            sections[canonical] = (sections[canonical] + " " + body_text).strip() \
                if canonical in sections else body_text

    full_text = "\n\n".join(content_parts)

    # Fallback for PDFs with no detectable heading structure (scans, preprints,
    # unusual templates): keep every non-boilerplate line rather than nothing.
    # Only when structure detection produced essentially nothing — a short but
    # correctly parsed paper must keep its sections.
    if not sections_ordered or len(full_text) < 200:
        salvaged = []
        for block in (b for blocks in pages_blocks for b in blocks):
            if _is_margin_matter(block, body_left):
                continue
            if any(_BOILERPLATE.search(text) for text, _, _ in block["lines"]):
                continue
            for text, _bold, _size in block["lines"]:
                if re.sub(r"\d+", "#", text) in running:
                    continue
                salvaged.append(text)
        salvaged_text = " ".join(salvaged).strip()
        if len(salvaged_text) > len(full_text):
            full_text = salvaged_text
            if not sections:
                sections = {"abstract": full_text[:4000]}
                sections_ordered = [{"heading": "Document", "text": full_text}]

    meta = dict(doc.metadata) if doc.metadata else {}
    title = str(meta.get("title", "") or "").strip() or _title_from_layout(doc) \
        or Path(pdf_path).stem
    author = str(meta.get("author", "") or "").strip()

    return {
        "full_text": full_text,
        "sections": sections,
        "sections_ordered": sections_ordered,
        "title": title,
        "author": author,
        "year": _publication_year(doc),
        "page_count": page_count,
        "char_count": len(full_text),
    }


# arXiv identifiers encode the original submission month: 1611.07004 is
# November 2016, regardless of which revision this PDF happens to be.
_ARXIV_ID = re.compile(r"arXiv:\s*(\d{2})(\d{2})\.\d{4,5}", re.I)

# "Received: 5 August 2024", "Published: 4 September 2024".
_DATED_LINE = re.compile(
    r"(?:published|received|accepted|revised)\b[^\n]{0,40}?(19|20)(\d{2})", re.I)

_COPYRIGHT_YEAR = re.compile(r"(?:©|\(c\)|copyright)\s*(19|20)(\d{2})", re.I)


def _publication_year(doc) -> Optional[int]:
    """Best estimate of when the paper was published, or None.

    Used to keep prior-art retrieval to work that could actually be prior art.
    Sources are tried most-trustworthy first; PDF metadata comes last because it
    records when the file was written, which can be a decade off — one arXiv
    preprint here reports 2026 for a 2016 paper because it was re-saved on
    download. That error direction is safe: a too-late year only widens the
    search back to its unfiltered behaviour, while a too-early one would hide
    genuine prior art.
    """
    upper = datetime.now().year + 1
    text = doc[0].get_text() if doc.page_count else ""

    match = _ARXIV_ID.search(text)
    if match:
        year = 2000 + int(match.group(1))
        if 2007 <= year <= upper:  # the YYMM.NNNNN scheme starts in 2007
            return year

    for pattern in (_DATED_LINE, _COPYRIGHT_YEAR):
        found = pattern.search(text)
        if found:
            year = int(found.group(1) + found.group(2))
            if 1970 <= year <= upper:
                return year

    stamp = (doc.metadata or {}).get("creationDate") or ""
    match = re.match(r"D:(\d{4})", stamp)
    if match:
        year = int(match.group(1))
        if 1970 <= year <= upper:
            return year

    return None


def _is_horizontal(line: dict) -> bool:
    """True for normal left-to-right text.

    PyMuPDF reports writing direction as a unit vector; horizontal text is
    (1, 0). Rotated text is nearly always a margin stamp rather than content.
    """
    direction = line.get("dir") or (1.0, 0.0)
    return abs(direction[0] - 1.0) < 0.01 and abs(direction[1]) < 0.01


def _title_from_layout(doc) -> str:
    """Largest horizontal text on page 1 — used when PDF metadata has no title.

    Rotated text is excluded. arXiv stamps its identifier down the left margin
    at a larger size than the title itself, so 'largest text wins' returned
    "arXiv:1611.07004v3 [cs.CV] 26 Nov 2018" instead of the paper's name.
    """
    if doc.page_count == 0:
        return ""
    spans = []
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if not _is_horizontal(line):
                continue
            for span in line["spans"]:
                text = span["text"].strip()
                if len(text) > 8:
                    spans.append((span["size"], text))
    if not spans:
        return ""
    top = max(s[0] for s in spans)
    return " ".join(t for s, t in spans if s >= top - 0.5)[:300]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks
