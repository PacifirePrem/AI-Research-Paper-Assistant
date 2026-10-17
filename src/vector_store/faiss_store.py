import faiss
import numpy as np


def build_index(embeddings):
    """Build a cosine-similarity index from normalized embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build an index without embeddings.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    # Inner product equals cosine similarity for normalized vectors.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


if __name__ == "__main__":
    embeddings = np.random.rand(10, 384).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = build_index(embeddings)

    print("Number of vectors in index:", index.ntotal)
