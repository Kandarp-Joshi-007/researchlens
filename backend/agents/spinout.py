from .base import get_llm, parse_score_response
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a venture capital analyst evaluating university research for spin-out company potential.
Assess market opportunity, team capability signals, technology readiness level (TRL), and competitive moat.
Respond with:
- SCORE: X/10
- RATIONALE: One paragraph explaining your reasoning
- Key points (as bullet list)"""),
    ("human", """Evaluate this research for spin-out/startup potential:

TITLE: {title}

ABSTRACT/INTRODUCTION:
{abstract}

DISCUSSION/CONCLUSIONS:
{discussion}

Score 1-10 where 10 = excellent spin-out potential (clear product path, large market, defensible IP).
"""),
])


def run(paper_data: dict) -> dict:
    llm = get_llm()
    chain = PROMPT | llm | StrOutputParser()
    response = chain.invoke({
        "title": paper_data.get("title", "Unknown"),
        "abstract": paper_data["sections"].get("abstract", paper_data["full_text"][:1500]),
        "discussion": paper_data["sections"].get("discussion", paper_data["sections"].get("conclusion", ""))[:1500],
    })
    result = parse_score_response(response)
    result["raw"] = response
    return result
