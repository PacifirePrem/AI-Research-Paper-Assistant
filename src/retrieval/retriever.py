from embeddings.embedder import create_query_embedding


def retrieve(query, index, chunks, k=5):
    """Retrieve the most similar chunks for a question."""
    if not query or not query.strip():
        return []

    if not chunks or index.ntotal == 0:
        return []

    if index.ntotal != len(chunks):
        raise ValueError(
            "The FAISS index and chunk list have different sizes. "
            "Reprocess the document to rebuild the index."
        )

    query_embedding = create_query_embedding(query)

    k = min(k, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

            print(f"Retrieval similarity: {score:.4f}")

    return results
