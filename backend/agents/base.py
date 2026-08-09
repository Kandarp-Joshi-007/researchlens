"""Shared LLM setup and structured scoring for the assessment agents."""

import logging
import os
from typing import List, Optional

import httpx
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

LLM_MODEL = os.getenv("RESEARCHLENS_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Ollama defaults to a 2048-token window regardless of what the model supports,
# which silently truncates long prompts. Sized for a 4 GB card: the KV cache at
# 8192 costs roughly 470 MB, which still leaves room for offloaded layers.
# Raise via RESEARCHLENS_NUM_CTX on a machine with more VRAM.
NUM_CTX = int(os.getenv("RESEARCHLENS_NUM_CTX", "8192"))

# Rough chars-per-token ratio for English prose, used to budget prompt context.
CHARS_PER_TOKEN = 4


class AgentVerdict(BaseModel):
    """One agent's assessment of a paper."""

    score: float = Field(ge=0, le=10, description="Score from 0 to 10")
    rationale: str = Field(description="One paragraph explaining the score")
    key_points: List[str] = Field(default_factory=list, description="3-5 short bullets")
    evidence: List[str] = Field(
        default_factory=list,
        description="Verbatim quotes from the paper supporting the assessment",
    )


# json_mode does not constrain the shape, so the prompt has to specify it.
FORMAT_INSTRUCTIONS = """
Respond with a single JSON object and nothing else, in exactly this shape:
{{
  "score": <number 0-10>,
  "rationale": "<one paragraph explaining the score>",
  "key_points": ["<short bullet>", "<short bullet>", "<short bullet>"],
  "evidence": ["<verbatim quote from the paper>", "<verbatim quote from the paper>"]
}}
Every entry in "evidence" must be copied word-for-word from the excerpts provided.
Do not invent quotes. If the excerpts do not support a confident judgement, say so
in the rationale and score accordingly."""


def get_llm(temperature: float = 0.0, num_ctx: Optional[int] = None) -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature=temperature,
        num_ctx=num_ctx or NUM_CTX,
    )


def context_budget_chars(reserve_tokens: int = 1200) -> int:
    """Chars of paper text that fit in the window, leaving room for prompt + reply."""
    return max(0, (NUM_CTX - reserve_tokens) * CHARS_PER_TOKEN)


_structured_method: Optional[str] = None


def _pick_structured_method() -> str:
    """Schema-constrained decoding needs Ollama >= 0.5.0; older servers get JSON mode.

    Older servers reject a schema object with 'cannot unmarshal object into Go
    struct field ChatRequest.format of type string'.
    """
    global _structured_method
    if _structured_method is not None:
        return _structured_method

    _structured_method = "json_mode"
    try:
        version = httpx.get(f"{OLLAMA_HOST}/api/version", timeout=3.0).json().get("version", "0")
        parts = [int(p) for p in version.split(".")[:3] if p.isdigit()]
        if parts >= [0, 5]:
            _structured_method = "json_schema"
        log.info("Ollama %s -> structured output via %s", version, _structured_method)
    except Exception as exc:  # server down or unreachable; JSON mode is the safe default
        log.warning("Could not read Ollama version (%s); using json_mode", exc)

    return _structured_method


def score_with_agent(prompt, variables: dict, temperature: float = 0.0,
                     attempts: int = 3) -> AgentVerdict:
    """Run a scoring prompt and return a validated verdict.

    Retries on malformed output. Raises if every attempt fails, so a broken
    agent surfaces as an error rather than a silent default score.
    """
    method = _pick_structured_method()
    last_error = None

    for attempt in range(attempts):
        # Nudge off a bad generation path on retry rather than repeating it.
        temp = temperature if attempt == 0 else max(temperature, 0.3)
        try:
            llm = get_llm(temperature=temp).with_structured_output(
                AgentVerdict, method=method
            )
            result = (prompt | llm).invoke(variables)
            if result is None:
                raise ValueError("model returned no parseable JSON")
            return result
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            log.warning("Structured output attempt %d/%d failed: %s",
                        attempt + 1, attempts, str(exc)[:200])

    raise RuntimeError(f"Agent failed to produce a valid verdict: {last_error}")
