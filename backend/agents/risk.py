from .base import get_llm, parse_score_response
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technology risk analyst. Evaluate research for commercialization risks.
Consider regulatory hurdles, ethical concerns, technology maturity, replication challenges, and market barriers.
Note: a HIGH risk score means HIGH RISK (bad for commercialization). A LOW score means LOW RISK (good).
Respond with:
- SCORE: X/10
- RATIONALE: One paragraph explaining your reasoning
- Key points (as bullet list)"""),
    ("human", """Evaluate the commercialization RISK of this research:

TITLE: {title}

ABSTRACT:
{abstract}

METHODS/DISCUSSION:
{methods}

Score 1-10 where 10 = extremely high risk (major regulatory/ethical/technical barriers), 1 = very low risk.
"""),
])


def run(paper_data: dict) -> dict:
    llm = get_llm()
    chain = PROMPT | llm | StrOutputParser()
    response = chain.invoke({
        "title": paper_data.get("title", "Unknown"),
        "abstract": paper_data["sections"].get("abstract", paper_data["full_text"][:1500]),
        "methods": paper_data["sections"].get("methods", paper_data["sections"].get("discussion", ""))[:1500],
    })
    result = parse_score_response(response)
    result["raw"] = response
    return result
