from .base import get_llm, parse_score_response
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a patent attorney evaluating research papers for patentability.
Assess novelty, non-obviousness, utility, and enablement.
Respond with:
- SCORE: X/10
- RATIONALE: One paragraph explaining your reasoning
- Key points (as bullet list)"""),
    ("human", """Evaluate this research for patentability:

TITLE: {title}

ABSTRACT/INTRODUCTION:
{abstract}

METHODS/RESULTS:
{methods}

Score this 1-10 where 10 = highly patentable (novel, non-obvious, clear utility).
"""),
])


def run(paper_data: dict) -> dict:
    llm = get_llm()
    chain = PROMPT | llm | StrOutputParser()
    response = chain.invoke({
        "title": paper_data.get("title", "Unknown"),
        "abstract": paper_data["sections"].get("abstract", paper_data["full_text"][:1500]),
        "methods": paper_data["sections"].get("methods", paper_data["sections"].get("results", ""))[:1500],
    })
    result = parse_score_response(response)
    result["raw"] = response
    return result
