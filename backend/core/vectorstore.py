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


def add_chunks(paper_id: int, chunks: list[str], metadata: dict):
    vs = get_vectorstore()
    ids = [f"paper_{paper_id}_chunk_{i}" for i in range(len(chunks))]
    metas = [{**metadata, "paper_id": paper_id, "chunk_index": i} for i in range(len(chunks))]
    vs.add_texts(texts=chunks, metadatas=metas, ids=ids)


def similarity_search(query: str, k: int = 4, paper_id: int = None) -> list[str]:
    vs = get_vectorstore()
    filter_dict = {"paper_id": paper_id} if paper_id else None
    docs = vs.similarity_search(query, k=k, filter=filter_dict)
    return [d.page_content for d in docs]


def delete_paper_chunks(paper_id: int):
    vs = get_vectorstore()
    result = vs.get(where={"paper_id": paper_id})
    if result and result.get("ids"):
        vs.delete(ids=result["ids"])
