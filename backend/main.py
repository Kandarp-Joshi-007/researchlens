import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import logging
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict

from backend.core.pdf_extractor import extract_text, chunk_text
from backend.core.database import (
    init_db, save_paper, save_score, save_summary, get_all_results,
    get_paper_scores, get_paper_summary, delete_paper, save_prior_art,
    get_prior_art, find_by_hash, create_run, update_run, get_latest_run,
    get_run_history, get_paper,
)
from backend.core.prior_art import find_similar_works
from backend.core.vectorstore import add_chunks, similarity_search, delete_paper_chunks
from backend.agents.base import get_llm
from backend.agents.definitions import AGENTS, overall_score, verdict_for
from backend.agents import runner
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ResearchLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger(__name__)

# A run with no progress for this long is presumed dead (backend restart).
STALE_RUN_SECONDS = int(os.getenv("RESEARCHLENS_STALE_RUN_SECONDS", "900"))


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    dest = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    digest = hashlib.sha256()
    with dest.open("wb") as f:
        while True:
            block = await file.read(1 << 20)
            if not block:
                break
            digest.update(block)
            f.write(block)
    sha256 = digest.hexdigest()

    # Identical content already scored: skip a multi-minute re-analysis.
    existing = find_by_hash(sha256)
    if existing:
        dest.unlink(missing_ok=True)
        return {
            "paper_id": existing["id"],
            "title": existing["title"],
            "pages": existing["page_count"],
            "duplicate": True,
            "overall": existing["overall"],
            "verdict": existing["verdict"],
        }

    try:
        paper_data = extract_text(str(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read PDF: {exc}")

    if paper_data["char_count"] < 200:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "No extractable text found — is this a scanned PDF?")

    paper_id = save_paper(file.filename, paper_data["title"], paper_data["author"],
                          paper_data["page_count"], filepath=str(dest), sha256=sha256)

    run_id = create_run(paper_id)
    background_tasks.add_task(_process_paper, paper_id, paper_data, run_id)

    return {"paper_id": paper_id, "title": paper_data["title"],
            "pages": paper_data["page_count"], "run_id": run_id}


def _process_paper(paper_id: int, paper_data: dict, run_id: int):
    try:
        update_run(run_id, "embedding")
        chunks = chunk_text(paper_data["full_text"])
        add_chunks(paper_id, chunks, {"title": paper_data["title"], "filename": paper_data.get("title", "")})

        update_run(run_id, "prior_art")
        works = find_similar_works(
            paper_data.get("title", ""),
            (paper_data.get("sections") or {}).get("abstract", ""),
        )
        save_prior_art(paper_id, works)

        scores = {}
        for spec in AGENTS:
            update_run(run_id, f"scoring:{spec['name']}")
            result = runner.run(spec, paper_data, paper_id=paper_id, prior_art=works)
            save_score(paper_id, spec["name"], result["score"],
                       result["rationale"], result["key_points"],
                       result["evidence"], run_id=run_id)
            scores[spec["name"]] = result["score"]

        overall = overall_score(scores)
        save_summary(paper_id, overall, verdict_for(overall))
        update_run(run_id, "done", done=True)

    except Exception as exc:
        log.exception("Analysis failed for paper %s", paper_id)
        update_run(run_id, "error", done=False, error=str(exc))


@app.get("/status/{paper_id}")
def get_status(paper_id: int):
    run = get_latest_run(paper_id)
    summary = get_paper_summary(paper_id)

    if run is None:
        if summary:
            return {"stage": "done", "done": True, "error": None,
                    "overall": summary["overall"]}
        return {"stage": "unknown", "done": False, "error": None}

    status = {
        "stage": run["stage"],
        "done": bool(run["done"]),
        "error": run["error"],
        "run_id": run["id"],
    }
    if summary:
        status["overall"] = summary["overall"]

    # A run left mid-flight by a backend restart is stale, not in progress.
    if not run["done"] and not run["error"]:
        age = _seconds_since(run["updated_at"])
        if age is not None and age > STALE_RUN_SECONDS:
            status["stale"] = True
            status["error"] = (
                f"No progress for {int(age // 60)} minutes — the backend may have "
                "restarted. Re-analyse to try again."
            )
    return status


def _seconds_since(timestamp: str):
    """Age of a SQLite UTC timestamp in seconds, or None if unparseable."""
    if not timestamp:
        return None
    try:
        moment = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None
    return (datetime.utcnow() - moment).total_seconds()


@app.post("/papers/{paper_id}/reanalyse")
def reanalyse(background_tasks: BackgroundTasks, paper_id: int):
    """Re-run the analysis, keeping previous runs for comparison."""
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    if not paper.get("filepath") or not Path(paper["filepath"]).exists():
        raise HTTPException(410, "Original PDF is no longer on disk")

    paper_data = extract_text(paper["filepath"])
    run_id = create_run(paper_id)
    background_tasks.add_task(_process_paper, paper_id, paper_data, run_id)
    return {"paper_id": paper_id, "run_id": run_id, "stage": "queued"}


@app.get("/papers/{paper_id}/history")
def paper_history(paper_id: int):
    return {"paper_id": paper_id, "runs": get_run_history(paper_id)}


@app.get("/results")
def list_results():
    return get_all_results()


@app.get("/results/{paper_id}")
def get_result(paper_id: int):
    scores = get_paper_scores(paper_id)
    if not scores:
        raise HTTPException(404, "Paper not found or not yet scored")
    summary = get_paper_summary(paper_id)
    overall = (summary or {}).get("overall")
    verdict = (summary or {}).get("verdict")
    return {"paper_id": paper_id, "scores": scores, "overall": overall,
            "verdict": verdict, "prior_art": get_prior_art(paper_id)}


@app.delete("/papers/{paper_id}")
def remove_paper(paper_id: int):
    filepath = delete_paper(paper_id)
    try:
        delete_paper_chunks(paper_id)
    except Exception:
        pass
    if filepath:
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass
    return {"deleted": paper_id}


_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant helping users understand a scientific paper.
Answer the question using ONLY the provided context excerpts from the paper.
Be concise and accurate. If the context doesn't contain enough information, say so."""),
    ("human", """Context from the paper:
{context}

Question: {question}
"""),
])


class QuestionRequest(BaseModel):
    question: str


@app.post("/query/{paper_id}")
def query_paper(paper_id: int, req: QuestionRequest):
    summary = get_paper_summary(paper_id)
    if not summary:
        raise HTTPException(404, "Paper not found or not yet processed")
    chunks = similarity_search(req.question, k=5, paper_id=paper_id)
    if not chunks:
        raise HTTPException(422, "No embeddings found for this paper")
    context = "\n\n---\n\n".join(chunks)
    chain = _QA_PROMPT | get_llm(temperature=0.1) | StrOutputParser()
    answer = chain.invoke({"context": context, "question": req.question})
    return {"answer": answer, "source_chunks": len(chunks)}
