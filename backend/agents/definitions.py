"""Declarative specs for the four assessment agents.

Personas, weights and prompts live here so that scoring changes happen in one
place rather than being duplicated across four near-identical modules.
"""

from langchain.prompts import ChatPromptTemplate

from .base import FORMAT_INSTRUCTIONS


def _prompt(system: str, human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", system + FORMAT_INSTRUCTIONS),
        ("human", human),
    ])


_EXCERPTS = """TITLE: {title}

EXCERPTS FROM THE PAPER:
{context}
"""

AGENTS = [
    {
        "name": "patentability",
        "label": "Patentability",
        "weight": 0.30,
        "inverted": False,
        "color": "#4C9BE8",
        # Queries used to pull the most relevant passages out of the paper.
        "queries": [
            "novel technical contribution, what is new compared to prior work",
            "specific method, architecture, algorithm or apparatus described",
            "technical advantage over existing approaches",
        ],
        # Novelty cannot be judged from the paper alone, so this agent is given
        # real published work to compare against.
        "uses_prior_art": True,
        "prompt": _prompt(
            """You are a patent attorney evaluating research papers for patentability.
Assess novelty, non-obviousness, utility, and enablement.
Score 1-10 where 10 = highly patentable (novel, non-obvious, clear utility,
sufficiently enabled). A survey or review paper that contributes no new
technical method should score low on novelty.
Judge novelty against the PRIOR ART listed below. Your rationale MUST begin with
one sentence naming the closest listed prior-art work and stating whether the
paper is meaningfully distinct from it; if closely related work already exists,
lower the score accordingly. If the prior-art list is empty, begin instead with
"No prior art was retrieved" and note that the novelty judgement is unverified.""",
            _EXCERPTS + """
PRIOR ART (similar published work retrieved from OpenAlex):
{prior_art}

Evaluate this research for patentability. Start your rationale by naming the
closest work in the PRIOR ART list above and comparing this paper to it.""",
        ),
    },
    {
        "name": "licensing",
        "label": "Licensing",
        "weight": 0.30,
        "inverted": False,
        "color": "#56C271",
        "queries": [
            "practical application, industrial use case, deployment",
            "performance improvement, benchmark results, measured benefit",
            "market relevance, adoption, existing commercial tools",
        ],
        "prompt": _prompt(
            """You are a technology licensing expert evaluating research for licensing potential.
Consider market size, ease of adoption, industry relevance, and the existing IP landscape.
Score 1-10 where 10 = excellent licensing prospects (large market, easy to adopt,
clear industry fit).""",
            _EXCERPTS + "\nEvaluate this research for technology licensing potential.",
        ),
    },
    {
        "name": "spinout",
        "label": "Spin-out",
        "weight": 0.25,
        "inverted": False,
        "color": "#E8A84C",
        "queries": [
            "technology readiness, prototype, experimental validation, real-world deployment",
            "scalability, cost, computational requirements",
            "competitive advantage, defensibility, comparison to alternatives",
        ],
        "prompt": _prompt(
            """You are a venture capital analyst evaluating university research for spin-out potential.
Assess market opportunity, technology readiness level (TRL), competitive moat, and
signals about team capability.
Score 1-10 where 10 = excellent spin-out potential (clear product path, large market,
defensible IP).""",
            _EXCERPTS + "\nEvaluate this research for spin-out/startup potential.",
        ),
    },
    {
        "name": "risk",
        "label": "Risk",
        "weight": 0.15,
        "inverted": True,  # high risk is bad, so it is inverted before weighting
        "color": "#E86B6B",
        "queries": [
            "limitations, weaknesses, threats to validity",
            "ethical concerns, privacy, bias, regulatory considerations",
            "reproducibility, data requirements, failure cases",
        ],
        "prompt": _prompt(
            """You are a technology risk analyst evaluating research for commercialization risk.
Consider regulatory hurdles, ethical concerns, technology maturity, reproducibility,
and market barriers.
Score 1-10 where 10 = extremely high risk (major regulatory, ethical or technical
barriers) and 1 = very low risk. A HIGH score means HIGH RISK, which is bad for
commercialization.""",
            _EXCERPTS + "\nEvaluate the commercialization RISK of this research.",
        ),
    },
]

AGENTS_BY_NAME = {a["name"]: a for a in AGENTS}


def overall_score(scores: dict) -> float:
    """Weighted overall score. Inverted dimensions are flipped before weighting."""
    total = 0.0
    for spec in AGENTS:
        if spec["name"] not in scores:
            continue
        value = scores[spec["name"]]
        if spec["inverted"]:
            value = 10 - value
        total += value * spec["weight"]
    return round(total, 2)


def verdict_for(overall: float) -> str:
    if overall >= 7.5:
        return "Strong Commercialisation Potential"
    if overall >= 5.0:
        return "Moderate Potential — Further Assessment Recommended"
    return "Limited Commercialisation Potential"
