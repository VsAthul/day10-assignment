import os
import sys
import chromadb
import ollama
from pypdf import PdfReader


def read_pdf(pdf_path: str = "./static/document.pdf") -> str:
    """Read and extract all text from a PDF file."""
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found at path: {pdf_path}")
        sys.exit(1)

    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    print(f"[INFO] Extracted {len(full_text)} characters from {len(reader.pages)} pages.")
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    print(f"[INFO] Created {len(chunks)} chunks.")
    return chunks


def store_in_chromadb(
    chunks: list[str],
    collection_name: str = "pdf_chunks",
    persist_dir: str = "./my_chroma_db",
    embedding_model: str = "qwen3-embedding:0.6b",
    batch_size: int = 50,  # safe batch size
):
    """Embed chunks in batches and store in ChromaDB."""

    client = chromadb.PersistentClient(path=persist_dir)

    # Remove old collection
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(name=collection_name)
        print(f"[INFO] Deleted existing collection '{collection_name}'.")

    collection = client.create_collection(name=collection_name)

    print(f"[INFO] Generating embeddings in batches (batch_size={batch_size})...")

    total_chunks = len(chunks)
    stored_count = 0

    # 🔹 Batch processing
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]

        # Single API call per batch
        res = ollama.embed(model=embedding_model, input=batch_chunks)
        embeddings = res.embeddings

        # Store batch
        for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
            idx = i + j
            collection.add(
                ids=[f"chunk_{idx}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"chunk_index": idx}],
            )

        stored_count += len(batch_chunks)
        print(f"[INFO] Stored {stored_count}/{total_chunks} chunks.")

    print(f"[SUCCESS] All chunks stored in ChromaDB collection '{collection_name}'.")


def run_ingestion(
    pdf_path: str = "./static/document.pdf",
    collection_name: str = "pdf_chunks",
    persist_dir: str = "./my_chroma_db",
    embedding_model: str = "qwen3-embedding:0.6b",
    chunk_size: int = 500,
    overlap: int = 100,
    batch_size: int = 50,
):
    """Full pipeline: read → chunk → embed → store."""
    print("=== Starting PDF Ingestion Pipeline ===")

    text = read_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    store_in_chromadb(
        chunks=chunks,
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        batch_size=batch_size,
    )

    print("=== Ingestion Complete ===")


if __name__ == "__main__":
    run_ingestion()