import fitz  # pymupdf
from pathlib import Path


def extract_text(pdf_path: str) -> dict:
    """Extract text and metadata from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    sections = {}
    current_section = "abstract"

    section_keywords = {
        "abstract": ["abstract"],
        "introduction": ["introduction", "background"],
        "methods": ["method", "methodology", "approach", "materials"],
        "results": ["result", "finding", "experiment"],
        "discussion": ["discussion", "analysis"],
        "conclusion": ["conclusion", "future work", "summary"],
        "references": ["reference", "bibliography"],
    }

    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text = str(block[4]).strip()
            if not text:
                continue
            lower = text.lower()
            for section, keywords in section_keywords.items():
                if any(lower.startswith(k) for k in keywords) and len(text) < 80:
                    current_section = section
                    break
            sections.setdefault(current_section, []).append(text)
            full_text += text + "\n"

    meta = dict(doc.metadata) if doc.metadata else {}
    title = str(meta.get("title", "") or Path(pdf_path).stem)
    author = str(meta.get("author", "") or "")
    page_count = int(doc.page_count)
    sections_plain = {k: " ".join(v) for k, v in sections.items()}
    full_text_plain = str(full_text)
    doc.close()

    return {
        "full_text": full_text_plain,
        "sections": sections_plain,
        "title": title,
        "author": author,
        "page_count": page_count,
        "char_count": len(full_text_plain),
    }


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
