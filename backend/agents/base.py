from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import re, json

LLM_MODEL = "qwen2.5:7b"


def get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=temperature)


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
