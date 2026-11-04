from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# One shared model for document and query embeddings.
model = SentenceTransformer(MODEL_NAME)


def create_embeddings(chunks):
    """Create normalized float32 embeddings for document chunks."""
    if not chunks:
        return np.empty((0, model.get_sentence_embedding_dimension()),
                        dtype=np.float32)

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)


def create_query_embedding(query):
    """Create a normalized embedding for a user's question."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(embedding, dtype=np.float32)


if __name__ == "__main__":
    chunks = [
        "Transformers are a neural network architecture.",
        "Attention mechanisms allow models to focus on important words.",
    ]

    embeddings = create_embeddings(chunks)

    print("Number of embeddings:", len(embeddings))
    print("Embedding vector size:", embeddings.shape[1])
    print("Data type:", embeddings.dtype)
    print("Vector norms:", np.linalg.norm(embeddings, axis=1))
