import os
import chromadb
import ollama
from chromadb.errors import NotFoundError


def embed_query(query: str, model: str = "qwen3-embedding:0.6b") -> list[float]:
    """Generate an embedding vector for the user query."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty or whitespace.")
    response = ollama.embed(model=model, input=query)
    return response.embeddings[0]


def collection_exists(collection_name: str = "pdf_chunks", persist_dir: str = "./my_chroma_db") -> bool:
    """Check whether a ChromaDB collection has been created."""
    try:
        if not os.path.exists(persist_dir):
            return False
        client = chromadb.PersistentClient(path=persist_dir)
        existing = [c.name for c in client.list_collections()]
        return collection_name in existing
    except Exception:
        return False


def retrieve_relevant_chunks(
    query: str,
    collection_name: str = "pdf_chunks",
    persist_dir: str = "./my_chroma_db",
    embedding_model: str = "qwen3-embedding:0.6b",
    top_k: int = 5,
) -> list[str]:
    """Embed a query and retrieve the top_k most relevant chunks from ChromaDB."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty or whitespace.")

    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(name=collection_name)
    except NotFoundError:
        raise RuntimeError(
            f"ChromaDB collection '{collection_name}' not found. "
            "Please run: python ingest.py"
        )

    query_embedding = embed_query(query, model=embedding_model)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results.get("documents", [[]])[0]