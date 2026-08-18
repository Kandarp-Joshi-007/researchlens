# ResearchLens

**Reads a research paper and scores its commercial potential — entirely on your own machine.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-150%20passing-brightgreen.svg)](tests/)

ResearchLens takes a PDF of an academic paper and produces a structured
commercialisation assessment: four scored dimensions, a rationale for each,
verbatim supporting quotes from the paper, and a novelty check against real
published prior art.

No cloud API, no API key, no per-paper cost. The language model runs locally
through [Ollama](https://ollama.com), so an unpublished manuscript never leaves
the machine — which for most technology transfer offices is the difference
between a tool they can use and one they can't.

![ResearchLens scoring a paper: four weighted dimensions on the left, and the
patentability agent's rationale, key points and verbatim evidence quotes on the
right](docs/analysis-view.png)

*Scoring [pix2pix](https://arxiv.org/abs/1611.07004) (Isola et al., 2016).
The patentability rationale opens by naming the closest retrieved prior art —
StackGAN, published the same year — and argues the distinction. Every score
carries quotes copied verbatim from the paper.*

---

## Why

A technology transfer office decides which research is worth patenting,
licensing, or spinning out. The bottleneck is expert reading time: assessing one
paper properly takes hours, and the supply of those hours is fixed. The result
isn't bad decisions — it's that most papers are never assessed at all.

ResearchLens doesn't replace that expert. It changes their starting point from a
blank page and a 38-page PDF to a structured brief they can check line by line.

## What it produces

| Dimension | Weight | Perspective | Question |
|---|---|---|---|
| **Patentability** | 30% | Patent attorney | Is it novel, non-obvious, and enabled? |
| **Licensing** | 30% | Licensing expert | Could a company license and sell this? |
| **Spin-out** | 25% | VC analyst | Could this become a startup? |
| **Risk** | 15% | Risk analyst | What could block it from reaching market? |

Risk is stored raw but **inverted before weighting**, so a high-risk paper is
penalised rather than rewarded:

```
Overall = Patentability × 0.30
        + Licensing     × 0.30
        + Spin-out      × 0.25
        + (10 − Risk)   × 0.15
```

```
≥ 7.5  →  Strong Commercialisation Potential
≥ 5.0  →  Moderate Potential — Further Assessment Recommended
<  5.0  →  Limited Commercialisation Potential
```

Weights are renormalised over whatever dimensions are present, so a partial
result can't silently deflate into a misleading verdict.

## Pipeline

1. **Upload & fingerprint** — the PDF is streamed to disk and hashed. Identical
   content already scored returns the saved result instead of repeating an
   eight-minute run.
2. **Structure-aware extraction** — headings detected, publisher boilerplate
   stripped, two-column layouts restored to reading order.
3. **Embedding** — overlapping 800-word chunks vectorised locally into ChromaDB.
4. **Prior-art lookup** — the title is searched against
   [OpenAlex](https://openalex.org), restricted to work published no later than
   the paper itself.
5. **Scoring** — four agents, each retrieving the passages relevant to its own
   concerns, each returning schema-validated structured output.
6. **Verdict** — weighted, banded, stored as a versioned run.
7. **Brief & Q&A** — export a one-page Markdown brief, or ask the paper
   questions answered only from its own text.

The single outbound request in the whole system is the OpenAlex lookup, and
`RESEARCHLENS_PRIOR_ART=0` disables even that.

## Quick start

Requires Python 3.9+ and [Ollama](https://ollama.com).

```bash
ollama serve                      # in its own terminal
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

pip install -r requirements.txt
bash run.sh
```

Then open **http://localhost:8501**.

`run.sh` binds to loopback by default. The API has no authentication, so
exposing it (`HOST=0.0.0.0 bash run.sh`) hands upload and delete to anyone who
can reach the machine — don't, without putting real auth in front of it.

### Running the pieces separately

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
python -m streamlit run frontend/app.py --server.port 8501
```

## Configuration

Copy `.env.example` to `.env`. Every setting is optional; the defaults suit a
local install. Loaded at import time, so changes need a restart.

| Variable | Default | Purpose |
|---|---|---|
| `RESEARCHLENS_MODEL` | `qwen2.5:7b` | Scoring and Q&A model |
| `RESEARCHLENS_NUM_CTX` | `8192` | Context window — Ollama otherwise defaults to 2048 and silently truncates |
| `RESEARCHLENS_DEEP_SAMPLES` | `3` | Samples per agent in deep mode |
| `RESEARCHLENS_MAX_IN_FLIGHT` | `6` | Analyses accepted at once; one runs, the rest queue |
| `RESEARCHLENS_PRIOR_ART` | `1` | Set `0` to stay fully offline |
| `RESEARCHLENS_MAX_UPLOAD_MB` | `50` | Upload size limit |

## Tests

```bash
python -m pytest        # 150 tests, ~13s
```

The suite stubs the language model and the network, so it runs anywhere without
a GPU or an internet connection. It includes a frontend suite driving the real
Streamlit script through `AppTest` with the backend stubbed.

## Notes from building it

A few problems where the obvious approach turned out to be wrong.

**Academic headings can't be detected by font size.** They're the same size as
body text and differ only by weight, so detection keys off the bold flag plus a
numbering pattern. The original keyword-matching version found 2 of 8 sections
and dumped 89% of the paper into the abstract bucket.

**A scoring failure must never produce a number.** The first implementation
parsed scores with a regex that took the first integer it found and defaulted to
`5.0`. Unparseable model output therefore produced a confident "Moderate
Potential" verdict out of nothing — invisible, and the worst possible failure
mode. Output is now schema-validated with retries, and a genuine failure raises.

**Small models ignore politely-worded instructions.** "Mention prior art if it
exists" was skipped entirely. Requiring the rationale to *open* by naming the
closest prior work made the model comply, and moved one paper's score from 6.0
to 4.0. Make requirements structurally mandatory, not requested.

**Agents run sequentially on purpose.** `qwen2.5:7b` is 4.7 GB and the
development GPU has 4 GB of VRAM, so it runs roughly half offloaded to CPU.
Concurrent inference would multiply the KV cache and make the whole run slower.
On a GPU that fits the model, this is worth revisiting.

**Background tasks are concurrent, which undid that.** The web framework runs
each upload's background work in its own thread, and the UI accepts multiple
files — so selecting four PDFs started four simultaneous analyses, exactly the
thrashing the sequential design existed to prevent. Analyses now queue behind a
lock, with a cap on how many may queue, because a waiting analysis occupies a
worker thread the API endpoints also draw on.

**Prior art has to predate the paper.** Retrieval sorted purely by relevance, so
for a 2016 paper all six results were published later — 2017 to 2020 — including
the one the agent named as "the closest prior art". Nothing published after a
paper can bear on its novelty. Adding a date filter took results postdating the
paper from six of six to zero of eight, and moved its patentability score from
6.0 to 8.0.

**The largest text on page one isn't always the title.** arXiv prints its
identifier down the left margin at 20pt, larger than the paper's own 14.3pt
title, so every arXiv preprint was titled `arXiv:1611.07004v3 [cs.CV] 26 Nov
2018`. Rotated text is now excluded from title detection.

**Evidence quotes are mandatory output.** They make a score checkable in seconds
and are the practical defence against a plausible-sounding invented rationale.

## Limitations

Worth being explicit about:

- **Not validated against outcomes.** The pipeline is reliable and scores are
  reproducible, but they haven't been checked against papers with known
  commercial results. That's the next piece of work, and no accuracy figure is
  claimed until it's done.
- **~8 minutes per paper** on a 4 GB laptop GPU. It's a batch tool.
- **Prior art is literature-only.** OpenAlex doesn't cover patent filings;
  PatentsView or EPO OPS would strengthen the patentability score considerably.
- **Prior-art matching is title-based.** Abstract or embedding matching would
  find closer work.
- **Temperature 0 buys repeatability, not accuracy.** Deep mode exists to expose
  the difference — it samples each dimension three times and reports the spread.
- **No authentication.** Correct for a single-machine local install; anything
  hosted needs real auth first.

This is decision support for a professional, not legal or investment advice.

## Stack

FastAPI · Streamlit · Ollama · ChromaDB · SQLite · PyMuPDF · LangChain · Pydantic

## Licence

MIT — see [LICENSE](LICENSE).
