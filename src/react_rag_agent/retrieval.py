from threading import Lock

from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .config import settings

_CACHE_LOCK = Lock()
_CACHED_VECTOR_STORE: Chroma | None = None


def get_vector_store() -> Chroma:
    global _CACHED_VECTOR_STORE

    with _CACHE_LOCK:
        if _CACHED_VECTOR_STORE is not None:
            return _CACHED_VECTOR_STORE

        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        _CACHED_VECTOR_STORE = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        return _CACHED_VECTOR_STORE


def reset_vector_store_cache() -> None:
    global _CACHED_VECTOR_STORE
    with _CACHE_LOCK:
        _CACHED_VECTOR_STORE = None


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    persist_dir = Path(settings.chroma_persist_dir)
    if not persist_dir.exists():
        print(f"Warning: Chroma persist directory not found at {persist_dir}")
        return []

    try:
        vector_store = get_vector_store()
        k = max(1, top_k or settings.top_k)
        results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=k,
        )
    except Exception as exc:
        print(f"Warning: Could not query vector store: {exc}")
        return []

    if not results:
        print("Warning: Vector store is empty or no relevant results were found.")
        return []

    formatted_results: list[dict] = []
    for document, score in results:
        formatted_results.append(
            {
                "content": document.page_content,
                "source": document.metadata.get("source", "unknown"),
                "page": document.metadata.get("page", "N/A"),
                "relevance_score": round(score, 3),
            }
        )

    return formatted_results
