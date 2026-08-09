import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict

from backend.core.pdf_extractor import extract_text, chunk_text
from backend.core.database import init_db, save_paper, save_score, save_summary, get_all_results, get_paper_scores, get_paper_summary, delete_paper
from backend.core.vectorstore import add_chunks, similarity_search, delete_paper_chunks
from backend.agents.base import get_llm
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import backend.agents.patentability as patent_agent
import backend.agents.licensing as license_agent
import backend.agents.spinout as spinout_agent
import backend.agents.risk as risk_agent

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ResearchLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track processing status in memory
_status: Dict[int, dict] = {}


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
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    paper_data = extract_text(str(dest))
    paper_id = save_paper(file.filename, paper_data["title"], paper_data["author"], paper_data["page_count"], filepath=str(dest))

    _status[paper_id] = {"stage": "embedding", "done": False, "error": None}
    background_tasks.add_task(_process_paper, paper_id, paper_data)

    return {"paper_id": paper_id, "title": paper_data["title"], "pages": paper_data["page_count"]}


def _process_paper(paper_id: int, paper_data: dict):
    try:
        _status[paper_id]["stage"] = "embedding"
        chunks = chunk_text(paper_data["full_text"])
        add_chunks(paper_id, chunks, {"title": paper_data["title"], "filename": paper_data.get("title", "")})

        agents = [
            ("patentability", patent_agent),
            ("licensing", license_agent),
            ("spinout", spinout_agent),
            ("risk", risk_agent),
        ]
        scores = {}
        for name, agent in agents:
            _status[paper_id]["stage"] = f"scoring:{name}"
            result = agent.run(paper_data)
            save_score(paper_id, name, result["score"], result["rationale"], result["key_points"])
            scores[name] = result["score"]

        # Overall = weighted average; risk is inverted (lower risk = better)
        overall = (
            scores["patentability"] * 0.30
            + scores["licensing"] * 0.30
            + scores["spinout"] * 0.25
            + (10 - scores["risk"]) * 0.15
        )

        if overall >= 7.5:
            verdict = "Strong Commercialisation Potential"
        elif overall >= 5.0:
            verdict = "Moderate Potential — Further Assessment Recommended"
        else:
            verdict = "Limited Commercialisation Potential"

        save_summary(paper_id, round(overall, 2), verdict)
        _status[paper_id] = {"stage": "done", "done": True, "error": None, "overall": round(overall, 2)}

    except Exception as e:
        _status[paper_id] = {"stage": "error", "done": False, "error": str(e)}


@app.get("/status/{paper_id}")
def get_status(paper_id: int):
    if paper_id in _status:
        return _status[paper_id]
    summary = get_paper_summary(paper_id)
    if summary:
        return {"stage": "done", "done": True, "error": None, "overall": summary["overall"]}
    return {"stage": "unknown", "done": False, "error": None}


@app.get("/results")
def list_results():
    return get_all_results()


@app.get("/results/{paper_id}")
def get_result(paper_id: int):
    scores = get_paper_scores(paper_id)
    if not scores:
        raise HTTPException(404, "Paper not found or not yet scored")
    summary = get_paper_summary(paper_id)
    overall = (summary or {}).get("overall") or _status.get(paper_id, {}).get("overall")
    verdict = (summary or {}).get("verdict")
    return {"paper_id": paper_id, "scores": scores, "overall": overall, "verdict": verdict}


@app.delete("/papers/{paper_id}")
def remove_paper(paper_id: int):
    filepath = delete_paper(paper_id)
    _status.pop(paper_id, None)
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
