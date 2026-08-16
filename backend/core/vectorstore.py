import os
import threading
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = str(Path(__file__).parent.parent.parent / "data" / "chroma")
EMBED_MODEL = os.getenv("RESEARCHLENS_EMBED_MODEL", "nomic-embed-text")

_stores = {}
_store_lock = threading.Lock()


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


def get_vectorstore(collection: str = "papers") -> Chroma:
    """One Chroma client per collection, reused for the process lifetime.

    Building a client per call opened a fresh persistent handle on the same
    directory on every search — twice per add_chunks alone — and Chroma rejects
    a second client on one path once its settings differ. The lock keeps the
    analysis thread and a concurrent API request from racing to create it.
    """
    store = _stores.get(collection)
    if store is not None:
        return store
    with _store_lock:
        if collection not in _stores:
            _stores[collection] = Chroma(
                collection_name=collection,
                embedding_function=get_embeddings(),
                persist_directory=CHROMA_DIR,
            )
    return _stores[collection]


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
    filter_dict = {"paper_id": paper_id} if paper_id is not None else None
    docs = vs.similarity_search(query, k=k, filter=filter_dict)
    return [d.page_content for d in docs]


def retrieve_for_queries(queries: list, paper_id: int, k_per_query: int = 2) -> list:
    """Chunks matching any of several queries, deduped and back in reading order.

    Retrieval returns chunks ranked by similarity; restoring document order
    keeps the excerpts readable as a narrative rather than a jumble.
    """
    vs = get_vectorstore()
    found = {}
    extra = []
    for query in queries:
        docs = vs.similarity_search(query, k=k_per_query, filter={"paper_id": paper_id})
        for doc in docs:
            index = doc.metadata.get("chunk_index")
            if index is None:
                # Written before chunk_index existed: no position to sort by, so
                # keep it at the end rather than colliding on a made-up index.
                if doc.page_content not in extra:
                    extra.append(doc.page_content)
                continue
            found.setdefault(index, doc.page_content)
    return [found[i] for i in sorted(found)] + extra


def delete_paper_chunks(paper_id: int):
    vs = get_vectorstore()
    result = vs.get(where={"paper_id": paper_id})
    if result and result.get("ids"):
        vs.delete(ids=result["ids"])
