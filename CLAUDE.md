# CLAUDE.md — ResearchLens

Working notes for future sessions. Covers the things that are **not** obvious from
reading the code: why decisions were made, what the hardware allows, and which
traps have already been hit.

For a plain-English product description see `PROJECT_EXPLAINED.txt`.

---

## What this is

A local-first tool that reads a research paper PDF and scores its commercial
potential across four dimensions (patentability, licensing, spin-out, risk),
then answers questions about the paper via RAG. FastAPI backend + Streamlit
frontend + Ollama. No data leaves the machine except optional OpenAlex prior-art
lookups.

## Commands

```bash
ollama serve                      # must be running first
bash run.sh                       # backend :8000 + frontend :8501
python -m pytest                  # 70 tests, ~10s, no GPU needed
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000   # backend only
python -m streamlit run frontend/app.py --server.port 8501        # frontend only
```

Tests stub the LLM and network, so they run offline and fast. **Run them before
committing** — they encode the bugs listed under "Traps" below.

## Layout

```
backend/
  main.py                 API routes, pipeline orchestration
  agents/
    base.py               LLM setup, AgentVerdict schema, structured-output retry
    definitions.py        THE FOUR AGENTS — personas, weights, prompts, queries
    runner.py             context assembly + sampling
  core/
    pdf_extractor.py      heading detection, boilerplate stripping
    database.py           SQLite, migrations, run versioning
    vectorstore.py        ChromaDB
    prior_art.py          OpenAlex client
    report.py             commercialisation brief
frontend/app.py           entire UI
tests/                    pytest suite
```

**To change scoring behaviour, edit `agents/definitions.py`.** Weights, prompts,
retrieval queries, inversion and verdict thresholds all live there. There is
deliberately no per-agent module — four near-identical files were consolidated
because every feature change required the same edit four times.

---

## Hardware reality (matters a lot)

Development machine: **RTX 3050 Laptop, 4 GB VRAM**, 16 GB RAM, Python 3.9.13.

`qwen2.5:7b` Q4_K_M is 4.7 GB, so it **does not fit in VRAM** — Ollama runs it
at roughly 46% CPU / 54% GPU, ~6.4 GB total footprint. Consequences:

- One agent takes **~2 minutes**. A full 4-agent analysis is **~8 minutes**.
- Deep mode (3 samples/agent) is **~25 minutes**. Treat as an overnight option.
- **Do not parallelise the agents here.** Concurrent slots multiply the KV cache
  and push more layers to CPU, making it slower. Revisit only on a GPU that fits
  the whole model. Sequential is intentional — see the comment in `_process_paper`.
- `NUM_CTX` is 8192 for the same reason (KV cache ≈ 470 MB). Raise via
  `RESEARCHLENS_NUM_CTX` on better hardware.

## Environment constraints

- **Python 3.9** — no `X | None` unions, no `match`, and **no backslashes inside
  f-string expressions** (that one is a syntax error and has already bitten once).
- **Ollama 0.4.0** — schema-constrained decoding (`format` as a JSON schema)
  landed in 0.5.0 and is **rejected** by this server with
  `cannot unmarshal object into Go struct field ChatRequest.format of type string`.
  `base.py` reads `/api/version` and picks `json_schema` or `json_mode`
  accordingly, so upgrading Ollama automatically improves reliability with no
  code change.
- Console here is cp1252. Em dashes in output often render as `?` — that is the
  terminal, not the data. Verify with `.encode('utf-8')` before "fixing" it.
  Read source files with `encoding='utf-8'` explicitly.

---

## Traps already hit (do not regress these)

1. **Section detection cannot key off font size.** Academic headings are the same
   size as body text and differ only by weight. Detection uses bold + a numbering
   pattern. Originally the extractor matched short lines starting with keywords,
   which found 2 of 8 sections and dumped 89% of the paper into `abstract`.
2. **Publisher boilerplate is multi-line and only its first line is marked.** It
   is filtered at *block* level, plus a geometric rule that drops narrow blocks
   sitting entirely left of the body column (MDPI's page-1 citation sidebar sorts
   into the middle of the introduction otherwise).
3. **Never let a scoring failure produce a number.** The old regex parser grabbed
   the first 0–10 integer anywhere in the response and defaulted to 5.0, silently
   yielding "Moderate" verdicts from nothing. `score_with_agent` retries then
   raises.
4. **The no-heading fallback must not fire on short-but-valid papers.** It once
   triggered under 500 chars and discarded correctly parsed structure. It now
   fires only when detection found nothing, and never replaces longer text with
   shorter.
5. **Soft prompt instructions get ignored by a 7B model.** "Say so explicitly if
   prior art exists" was skipped entirely; requiring the rationale to *open* by
   naming the closest work made it comply (and moved the score 6.0 → 4.0). If a
   7B agent is not doing something, make it structurally mandatory rather than
   politely requested.
6. **Chart colours are validated, not chosen by eye.** The original green/amber
   pair was ΔE 4.1 under protanopia — indistinguishable. Current palette in
   `AGENT_COLORS` passes in light and dark. If you change it, re-validate; do not
   eyeball it.

## Design decisions worth keeping

- **Agents retrieve their own context.** Each spec carries `queries`; the runner
  pulls matching chunks (deduped, restored to reading order) plus the abstract.
  Measured overlap between agents is 11–25%, so they genuinely read different
  things. Falls back to canonical sections when embeddings are unavailable.
- **Evidence quotes are mandatory output.** They make a score traceable, and
  spot-checking confirmed they are verbatim. This is the main defence against a
  plausible-sounding invented rationale.
- **Runs are versioned.** `scores.run_id` means re-analysis replaces rather than
  stacks; `get_paper_scores` returns only the latest run, `/papers/{id}/history`
  returns all. Rows predating this have `run_id NULL` and are still readable.
- **Prior art fails soft.** Any OpenAlex error returns `[]` and the analysis
  continues; set `RESEARCHLENS_PRIOR_ART=0` to stay fully offline.
- **Temperature 0 buys repeatability, not accuracy.** Deep mode exists to expose
  that difference. Do not describe deterministic scores as "reliable".

---

## State of the data

`data/results.db` holds papers 8 and 9 (the same PDF, analysed before and after
the fixes) — useful as a before/after reference. Paper 8's scores predate run
tracking and have `run_id NULL`; paper 9 has a real run in history.

Both were backfilled with SHA-256 hashes so dedupe recognises them.

## Known gaps / next steps

- **Prior art is title-search only.** Abstract-based or embedding-based matching
  would find closer work. The `abstract` parameter of `find_similar_works` is
  accepted but not yet used.
- **No patent-database lookup.** OpenAlex covers published literature; PatentsView
  or EPO OPS would make the patentability score much stronger.
- **Q&A retrieval is not reranked** — top-5 by cosine similarity only.
- **No auth.** CORS is locked to localhost:8501; anything hosted needs real auth.
- **Streamlit reruns the whole script** outside the progress fragment. Fine at
  this scale; would need caching for a large library.
- Upgrading Ollama past 0.5.0 is the single cheapest reliability win available.
