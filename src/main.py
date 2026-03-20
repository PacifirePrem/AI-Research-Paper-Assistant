from ingestion.pdf_loader import load_pdf
from preprocessing.chunk_text import chunk_text
from embeddings.embedder import create_embeddings
from vector_store.faiss_store import build_index
from retrieval.retriever import retrieve
from generation.rag_pipeline import generate_answer


def build_pipeline(pdf_path):

    text = load_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    return chunks, index


def answer_query(query, chunks, index):

    retrieved_chunks = retrieve(query, index, chunks)

    answer = generate_answer(query, retrieved_chunks)

    return answer