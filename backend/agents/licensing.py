from .base import get_llm, parse_score_response
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technology licensing expert evaluating research for licensing potential.
Consider market size, ease of adoption, industry relevance, existing IP landscape.
Respond with:
- SCORE: X/10
- RATIONALE: One paragraph explaining your reasoning
- Key points (as bullet list)"""),
    ("human", """Evaluate this research for technology licensing potential:

TITLE: {title}

ABSTRACT:
{abstract}

RESULTS/CONCLUSIONS:
{conclusions}

Score 1-10 where 10 = excellent licensing prospects (large market, easy to adopt, clear industry fit).
"""),
])


def run(paper_data: dict) -> dict:
    llm = get_llm()
    chain = PROMPT | llm | StrOutputParser()
    response = chain.invoke({
        "title": paper_data.get("title", "Unknown"),
        "abstract": paper_data["sections"].get("abstract", paper_data["full_text"][:1500]),
        "conclusions": paper_data["sections"].get("conclusion", paper_data["sections"].get("results", ""))[:1500],
    })
    result = parse_score_response(response)
    result["raw"] = response
    return result
