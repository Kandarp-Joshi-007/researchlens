import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import logging
import re
import sqlite3
import threading
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.core.pdf_extractor import extract_text, chunk_text
from backend.core.database import (
    init_db, save_paper, save_score, save_summary, get_all_results,
    get_paper_scores, get_paper_summary, delete_paper, save_prior_art,
    get_prior_art, find_by_hash, create_run, update_run, get_latest_run,
    get_run_history, get_paper,
)
from backend.core.prior_art import find_similar_works
from backend.core.report import build_markdown
from backend.core.vectorstore import add_chunks, similarity_search, delete_paper_chunks
from backend.agents.base import LLM_MODEL, NUM_CTX, get_llm
from backend.agents.definitions import AGENTS, overall_score, verdict_for
from backend.agents import runner
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def configure_logging():
    """Make the application's own log lines visible.

    Uvicorn configures only its own loggers and leaves the root at WARNING, so
    without this every progress line from the analysis pipeline is dropped and
    an eight-minute run reports nothing.
    """
    level = getattr(logging, os.getenv("RESEARCHLENS_LOG_LEVEL", "INFO").upper(),
                    logging.INFO)
    root = logging.getLogger("backend")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(handler)
    # Third-party libraries are chatty at INFO; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    log.info("ResearchLens API ready (model=%s, ctx=%s)", LLM_MODEL, NUM_CTX)
    yield


app = FastAPI(title="ResearchLens API", version="1.1.0", lifespan=lifespan)

# Local-first by default: the Streamlit UI is the only intended caller.
# Override with a comma-separated RESEARCHLENS_ALLOWED_ORIGINS if you host this.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "RESEARCHLENS_ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

log = logging.getLogger(__name__)

# A run with no progress for this long is presumed dead (backend restart).
STALE_RUN_SECONDS = int(os.getenv("RESEARCHLENS_STALE_RUN_SECONDS", "900"))

# Samples per agent in deep-analysis mode, used to report score spread.
DEEP_SAMPLES = int(os.getenv("RESEARCHLENS_DEEP_SAMPLES", "3"))


MAX_UPLOAD_BYTES = int(os.getenv("RESEARCHLENS_MAX_UPLOAD_MB", "50")) * 1024 * 1024


def safe_filename(name: str) -> str:
    """Reduce an uploaded filename to a harmless basename.

    A client controls this string entirely. Left alone, 'a/../../x.pdf' escapes
    the uploads directory and an absolute path raises OSError on Windows.

    Truncation keeps the extension: shortening the whole string used to drop the
    '.pdf' off a long name, which the caller then rejected as "not a PDF".
    """
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w.\- ]", "_", name).strip(". ")
    if not name:
        return "upload.pdf"

    stem, dot, ext = name.rpartition(".")
    if not dot or len(ext) > 10:  # no usable extension; truncate whole
        return name[:120]
    return (stem[:120 - len(ext) - 1] + "." + ext) if len(name) > 120 else name


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                       deep: bool = False):
    """Upload a PDF and start analysis.

    deep=true scores each dimension three times and reports the spread, at
    roughly three times the runtime.
    """
    original_name = safe_filename(file.filename)
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    dest = UPLOAD_DIR / f"{uuid.uuid4()}_{original_name}"
    digest = hashlib.sha256()
    size = 0
    try:
        with dest.open("wb") as f:
            while True:
                block = await file.read(1 << 20)
                if not block:
                    break
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                digest.update(block)
                f.write(block)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not store upload: {exc}")

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty")

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
        log.info("Rejected upload %s: %s", original_name, exc)
        # The message reaches the browser, so it must not carry the server-side
        # storage path that PyMuPDF puts in its errors.
        raise HTTPException(400, f"Could not read PDF: {_scrub_paths(exc)}")

    if paper_data["char_count"] < 200:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "No extractable text found — is this a scanned PDF?")

    # Reserved before the paper row exists so a refusal leaves nothing behind.
    try:
        _reserve_slot()
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    paper_id = save_paper(original_name, paper_data["title"], paper_data["author"],
                          paper_data["page_count"], filepath=str(dest), sha256=sha256)

    run_id = create_run(paper_id)
    background_tasks.add_task(_process_paper, paper_id, paper_data, run_id,
                              DEEP_SAMPLES if deep else 1)

    return {"paper_id": paper_id, "title": paper_data["title"],
            "pages": paper_data["page_count"], "run_id": run_id, "deep": deep}


# One analysis at a time, process-wide. FastAPI runs each upload's background
# task in its own threadpool worker, so selecting four PDFs in the sidebar
# started four analyses at once — measured peak 4 — each loading the same
# partly-CPU-offloaded model. That multiplies the KV cache and thrashes exactly
# the way the sequential-agent loop below exists to avoid. Papers queue instead.
_ANALYSIS_LOCK = threading.Lock()

# A queued analysis blocks the threadpool worker it waits in, and the sync API
# endpoints draw on that same pool. Admission is capped so a bulk upload cannot
# starve /status and /results and hang the UI; past the cap the upload is
# refused with 429 rather than silently queued behind an hour of work.
MAX_IN_FLIGHT = int(os.getenv("RESEARCHLENS_MAX_IN_FLIGHT", "6"))
_ADMISSION = threading.Semaphore(MAX_IN_FLIGHT)


def _reserve_slot():
    """Claim a place in the analysis queue, or raise 429 if it is full."""
    if not _ADMISSION.acquire(blocking=False):
        raise HTTPException(
            429,
            f"{MAX_IN_FLIGHT} analyses are already queued. Each takes several "
            "minutes; wait for one to finish and try again.",
        )


def _process_paper(paper_id: int, paper_data: dict, run_id: int, samples: int = 1):
    """Run one analysis, waiting for the single analysis slot to free up.

    Callers must have reserved an admission slot first; it is released here.
    """
    try:
        if not _ANALYSIS_LOCK.acquire(blocking=False):
            log.info("Analysis of paper %s is queued behind another run", paper_id)
            # Waiting happens in slices so the run's updated_at keeps moving;
            # otherwise a paper queued behind two others trips the stale-run
            # detector before it has even started.
            while not _ANALYSIS_LOCK.acquire(timeout=30):
                if not get_paper(paper_id):
                    log.info("Queued paper %s was deleted; dropping run %s",
                             paper_id, run_id)
                    return
                update_run(run_id, "queued")
        try:
            _run_analysis(paper_id, paper_data, run_id, samples)
        finally:
            _ANALYSIS_LOCK.release()
    finally:
        _ADMISSION.release()


def _run_analysis(paper_id: int, paper_data: dict, run_id: int, samples: int = 1):
    try:
        update_run(run_id, "embedding")
        chunks = chunk_text(paper_data["full_text"])
        try:
            add_chunks(paper_id, chunks,
                       {"title": paper_data["title"],
                        "filename": paper_data.get("title", "")})
        except Exception as exc:
            # Without embeddings the agents fall back to whole sections, which
            # is worse but still a usable analysis. Q&A stays unavailable.
            log.warning("Embedding failed for paper %s (%s); scoring from sections",
                        paper_id, exc)

        update_run(run_id, "prior_art")
        works = find_similar_works(
            paper_data.get("title", ""),
            (paper_data.get("sections") or {}).get("abstract", ""),
            before_year=paper_data.get("year"),
        )
        save_prior_art(paper_id, works)

        # Agents run sequentially on purpose. Running them concurrently only
        # helps when the whole model fits in VRAM with room for several KV
        # caches; on a 4 GB card qwen2.5:7b is already ~46% offloaded to CPU,
        # so parallel slots would multiply the cache and thrash. Raise
        # OLLAMA_NUM_PARALLEL and revisit this on a larger GPU.
        scores = {}
        for spec in AGENTS:
            # The user can delete a paper while its analysis is still running.
            # Stop quietly rather than writing rows for a paper that is gone.
            if not get_paper(paper_id):
                log.info("Paper %s deleted mid-analysis; abandoning run %s",
                         paper_id, run_id)
                return

            update_run(run_id, f"scoring:{spec['name']}")
            result = runner.run(spec, paper_data, paper_id=paper_id,
                                prior_art=works, samples=samples)

            # Re-check after the agent returns, not just before it starts: an
            # agent takes minutes, and a delete landing inside that window made
            # the insert below fail on the foreign key and reported a normal
            # user action as a crashed analysis.
            if not get_paper(paper_id):
                log.info("Paper %s deleted while %s was scoring; abandoning run %s",
                         paper_id, spec["name"], run_id)
                return

            save_score(paper_id, spec["name"], result["score"],
                       result["rationale"], result["key_points"],
                       result["evidence"], run_id=run_id,
                       score_min=result["score_min"], score_max=result["score_max"],
                       samples=result["samples"])
            scores[spec["name"]] = result["score"]

        if not get_paper(paper_id):
            log.info("Paper %s deleted mid-analysis; discarding results", paper_id)
            return

        overall = overall_score(scores)
        save_summary(paper_id, overall, verdict_for(overall))
        update_run(run_id, "done", done=True)

    except sqlite3.IntegrityError:
        # The re-checks above leave a microsecond window in which a delete can
        # still land between the check and the insert. Losing that race is the
        # user getting what they asked for, not a failure worth reporting.
        if not get_paper(paper_id):
            log.info("Paper %s deleted mid-analysis; abandoning run %s",
                     paper_id, run_id)
            return
        raise
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
def reanalyse(background_tasks: BackgroundTasks, paper_id: int, deep: bool = False):
    """Re-run the analysis, keeping previous runs for comparison."""
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    if not paper.get("filepath") or not Path(paper["filepath"]).exists():
        raise HTTPException(410, "Original PDF is no longer on disk")

    try:
        paper_data = extract_text(paper["filepath"])
    except Exception as exc:
        # The stored file can rot between runs (truncated, replaced, corrupted).
        raise HTTPException(422, f"Could not re-read the stored PDF: {_short(exc)}")

    _reserve_slot()
    run_id = create_run(paper_id)
    background_tasks.add_task(_process_paper, paper_id, paper_data, run_id,
                              DEEP_SAMPLES if deep else 1)
    return {"paper_id": paper_id, "run_id": run_id, "stage": "queued", "deep": deep}


@app.get("/papers/{paper_id}/report")
def paper_report(paper_id: int):
    """Commercialisation brief as Markdown."""
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    scores = get_paper_scores(paper_id)
    if not scores:
        raise HTTPException(409, "Paper has not been scored yet")

    markdown = build_markdown(paper, scores, get_paper_summary(paper_id),
                              get_prior_art(paper_id))
    return {"paper_id": paper_id, "filename": _report_filename(paper),
            "markdown": markdown}


def _report_filename(paper: dict) -> str:
    stem = re.sub(r"[^\w\s-]", "", paper.get("title") or "report").strip()
    stem = re.sub(r"\s+", "-", stem)[:60] or "report"
    return f"{stem}-commercialisation-brief.md"


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
    if not get_paper(paper_id):
        raise HTTPException(404, "Paper not found")
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
    question: str = Field(min_length=1, max_length=2000)


@app.post("/query/{paper_id}")
def query_paper(paper_id: int, req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(422, "Question is empty")

    summary = get_paper_summary(paper_id)
    if not summary:
        raise HTTPException(404, "Paper not found or not yet processed")

    # Retrieval needs the embedding model, so Ollama being down surfaces here
    # first. Report it as a dependency failure rather than a bare 500.
    try:
        chunks = similarity_search(question, k=5, paper_id=paper_id)
    except Exception as exc:
        log.warning("Retrieval failed for paper %s: %s", paper_id, exc)
        raise HTTPException(503, f"Search backend unavailable: {_short(exc)}")

    if not chunks:
        raise HTTPException(422, "No embeddings found for this paper")

    context = "\n\n---\n\n".join(chunks)
    chain = _QA_PROMPT | get_llm(temperature=0.1) | StrOutputParser()
    try:
        answer = chain.invoke({"context": context, "question": question})
    except Exception as exc:
        log.warning("Q&A generation failed for paper %s: %s", paper_id, exc)
        raise HTTPException(
            503,
            f"Could not reach the language model — is `ollama serve` running? "
            f"({_short(exc)})",
        )
    return {"answer": answer, "source_chunks": len(chunks)}


def _short(exc: Exception, limit: int = 160) -> str:
    return str(exc).replace("\n", " ")[:limit]


# Absolute paths, Windows or POSIX, as they appear inside library error strings.
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s'\"]*[\\/][^\s'\"]*")


def _scrub_paths(exc: Exception, limit: int = 160) -> str:
    """An error message with server filesystem paths replaced by the basename.

    PyMuPDF reports 'Failed to open file <full path>'. Echoing that to the
    client publishes the install location and the uploads directory layout.
    """
    def basename(match):
        return match.group(0).replace("\\", "/").rstrip("/").split("/")[-1]

    return _PATH_RE.sub(basename, str(exc)).replace("\n", " ")[:limit]
