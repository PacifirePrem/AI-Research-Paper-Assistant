import os
import joblib

from ingestion.pdf_loader import load_pdf
from preprocessing.chunk_text import chunk_text
from embeddings.embedder import create_embeddings
from vector_store.faiss_store import build_index
from retrieval.retriever import retrieve
from generation.rag_pipeline import generate_answer


# Save paths
INDEX_PATH = "data/faiss_index.index"
CHUNKS_PATH = "data/chunks.pkl"


def build_pipeline(pdf_path):

    # If already processed → load
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):

        print("Loading saved index and chunks...")

        index = joblib.load(INDEX_PATH)
        chunks = joblib.load(CHUNKS_PATH)

        return chunks, index

    print("Processing new document...")

    text = load_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    # Save
    os.makedirs("data", exist_ok=True)
    joblib.dump(index, INDEX_PATH)
    joblib.dump(chunks, CHUNKS_PATH)

    print("Saved index and chunks!")

    return chunks, index


def answer_query(query, chunks, index):

    retrieved_chunks = retrieve(query, index, chunks)

    answer = generate_answer(query, retrieved_chunks)

    return answer