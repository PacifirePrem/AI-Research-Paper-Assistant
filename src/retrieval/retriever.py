from sentence_transformers import SentenceTransformer
import numpy as np


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, index, chunks, k=5):

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Search FAISS index
    distances, indices = index.search(np.array(query_embedding), k)

    # Return top chunks
    results = [chunks[i] for i in indices[0]]

    return results


if __name__ == "__main__":

    # Example chunks
    chunks = [
        "Transformers use attention mechanisms.",
        "BERT is a language representation model.",
        "Neural networks learn hierarchical features.",
        "Self-attention helps models focus on important tokens."
    ]

    # Fake embeddings for testing
    embeddings = model.encode(chunks)

    import faiss

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    query = "What is attention in transformers?"

    results = retrieve(query, index, chunks)

    print("\nTop relevant chunks:\n")

    for r in results:
        print("-", r)