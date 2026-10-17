import hashlib
from pathlib import Path

import joblib

from ingestion.pdf_loader import load_pdf
from preprocessing.chunk_text import chunk_text
from embeddings.embedder import create_embeddings
from vector_store.faiss_store import build_index
from retrieval.retriever import retrieve
from generation.rag_pipeline import generate_answer


# A new cache version prevents reuse of indexes built with old chunk formats.
CACHE_DIR = Path("data") / "indexes_v3"


def get_document_hash(pdf_path):
    digest = hashlib.sha256()

    with open(pdf_path, "rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_pipeline(pdf_path):
    document_hash = get_document_hash(pdf_path)

    index_path = CACHE_DIR / f"{document_hash}.index"
    chunks_path = CACHE_DIR / f"{document_hash}.chunks.pkl"

    if index_path.exists() and chunks_path.exists():
        print(f"Loading cached document: {document_hash[:12]}...")

        index = joblib.load(index_path)
        chunks = joblib.load(chunks_path)

        # Prevent accidental use of incompatible cache files.
        if index.ntotal != len(chunks):
            raise ValueError(
                "Cached index and chunks do not match. "
                "Use a fresh cache directory."
            )

        return chunks, index

    print(f"Processing document: {document_hash[:12]}...")

    pages = load_pdf(pdf_path)
    chunks = chunk_text(pages)

    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    # Embed text only; keep page metadata in the chunk records.
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(chunk_texts)
    index = build_index(embeddings)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(index, index_path)
    joblib.dump(chunks, chunks_path)

    print(f"Saved document cache: {document_hash[:12]}...")
    return chunks, index


def answer_query(query, chunks, index):
    retrieved_chunks = retrieve(query, index, chunks)

    if not retrieved_chunks:
        return (
            "I couldn't find relevant passages in the extracted paper text.",
            [],
        )

    answer = generate_answer(query, retrieved_chunks)

    source_pages = list(dict.fromkeys(
        chunk["page"] for chunk in retrieved_chunks
    ))

    return answer, retrieved_chunks