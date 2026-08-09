import os
import re

from langchain_ollama import ChatOllama

LLM_MODEL = os.getenv("RESEARCHLENS_MODEL", "qwen2.5:7b")

# Ollama defaults to a 2048-token window regardless of what the model supports,
# which silently truncates long prompts. Sized for a 4 GB card: the KV cache at
# 8192 costs roughly 470 MB, which still leaves room for offloaded layers.
# Raise via RESEARCHLENS_NUM_CTX on a machine with more VRAM.
NUM_CTX = int(os.getenv("RESEARCHLENS_NUM_CTX", "8192"))

# Rough chars-per-token ratio for English prose, used to budget prompt context.
CHARS_PER_TOKEN = 4


def get_llm(temperature: float = 0.0, num_ctx: int = None) -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature=temperature,
        num_ctx=num_ctx or NUM_CTX,
    )


def context_budget_chars(reserve_tokens: int = 1200) -> int:
    """Chars of paper text that fit in the window, leaving room for prompt + reply."""
    return max(0, (NUM_CTX - reserve_tokens) * CHARS_PER_TOKEN)


def parse_score_response(text: str) -> dict:
    """Extract score, rationale, and key_points from LLM response."""
    score = None
    score_match = re.search(r"(?:score|SCORE)[:\s]+([0-9]{1,2}(?:\.[0-9])?)\s*/?\s*10", text, re.I)
    if score_match:
        score = float(score_match.group(1))

    if score is None:
        # Only accept standalone numbers that are plausible scores (0-10)
        nums = re.findall(r"(?<!\d)([0-9](?:\.[0-9])?|10)(?!\d)", text)
        score_candidates = [float(n) for n in nums if 0.0 <= float(n) <= 10.0]
        if score_candidates:
            score = score_candidates[0]

    score = max(0.0, min(10.0, score or 5.0))

    key_points = []
    bullet_matches = re.findall(r"[-•*]\s+(.+)", text)
    numbered_matches = re.findall(r"\d+\.\s+(.+)", text)
    key_points = (bullet_matches or numbered_matches)[:5]

    rationale_match = re.search(
        r"(?:rationale|reasoning|justification|explanation)[:\s]+(.+?)(?:\n\n|\Z)",
        text, re.I | re.S
    )
    rationale = rationale_match.group(1).strip() if rationale_match else text[:500]

    return {"score": score, "rationale": rationale, "key_points": key_points}
