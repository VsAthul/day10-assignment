import os
import sys
import chromadb
import ollama
from pypdf import PdfReader


def read_pdf(pdf_path: str = "./static/gold_loan_pdf.pdf") -> str:
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
    """Split text into overlapping chunks for better context retrieval."""
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


def generate_embedding(text: str, model: str = "qwen3-embedding:0.6b") -> list[float]:
    """Generate an embedding vector for a given text using Ollama."""
    response = ollama.embed(model=model, input=text)
    return response.embeddings[0]


def store_in_chromadb(
    chunks: list[str],
    collection_name: str = "pdf_chunks",
    persist_dir: str = "./my_chroma_db",
    embedding_model: str = "qwen3-embedding:0.6b",
):
    """Embed each chunk and store in a ChromaDB persistent collection."""
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if it exists to avoid duplicates on re-ingestion
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(name=collection_name)
        print(f"[INFO] Deleted existing collection '{collection_name}'.")

    collection = client.create_collection(name=collection_name)

    print(f"[INFO] Generating embeddings and storing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk, model=embedding_model)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"chunk_index": i}],
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"[INFO] Stored {i + 1}/{len(chunks)} chunks.")

    print(f"[SUCCESS] All chunks stored in ChromaDB collection '{collection_name}'.")


def run_ingestion(
    pdf_path: str = "./static/gold_loan_pdf.pdf",
    collection_name: str = "pdf_chunks",
    persist_dir: str = "./my_chroma_db",
    embedding_model: str = "qwen3-embedding:0.6b",
    chunk_size: int = 500,
    overlap: int = 100,
):
    """Full ingestion pipeline: read → chunk → embed → store."""
    print("=== Starting PDF Ingestion Pipeline ===")
    text = read_pdf(pdf_path=pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    store_in_chromadb(
        chunks,
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
    )
    print("=== Ingestion Complete ===")


if __name__ == "__main__":
    run_ingestion()
