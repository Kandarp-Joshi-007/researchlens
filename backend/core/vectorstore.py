import chromadb
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = str(Path(__file__).parent.parent.parent / "data" / "chroma")
EMBED_MODEL = "nomic-embed-text"


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


def get_vectorstore(collection: str = "papers") -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def add_chunks(paper_id: int, chunks: list, metadata: dict):
    """Replace this paper's embeddings.

    Re-analysis can produce fewer chunks than the previous run (a better
    extractor drops boilerplate). Ids are positional, so writing without
    clearing first leaves the tail of the old run behind and retrieval then
    mixes stale text with new.
    """
    vs = get_vectorstore()
    delete_paper_chunks(paper_id)
    if not chunks:
        return
    ids = [f"paper_{paper_id}_chunk_{i}" for i in range(len(chunks))]
    metas = [{**metadata, "paper_id": paper_id, "chunk_index": i} for i in range(len(chunks))]
    vs.add_texts(texts=chunks, metadatas=metas, ids=ids)


def similarity_search(query: str, k: int = 4, paper_id: int = None) -> list:
    vs = get_vectorstore()
    filter_dict = {"paper_id": paper_id} if paper_id else None
    docs = vs.similarity_search(query, k=k, filter=filter_dict)
    return [d.page_content for d in docs]


def retrieve_for_queries(queries: list, paper_id: int, k_per_query: int = 2) -> list:
    """Chunks matching any of several queries, deduped and back in reading order.

    Retrieval returns chunks ranked by similarity; restoring document order
    keeps the excerpts readable as a narrative rather than a jumble.
    """
    vs = get_vectorstore()
    found = {}
    for query in queries:
        docs = vs.similarity_search(query, k=k_per_query, filter={"paper_id": paper_id})
        for doc in docs:
            index = doc.metadata.get("chunk_index", len(found))
            found.setdefault(index, doc.page_content)
    return [found[i] for i in sorted(found)]


def delete_paper_chunks(paper_id: int):
    vs = get_vectorstore()
    result = vs.get(where={"paper_id": paper_id})
    if result and result.get("ids"):
        vs.delete(ids=result["ids"])
