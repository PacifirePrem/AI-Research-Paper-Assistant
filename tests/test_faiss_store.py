import faiss
import numpy as np
import pytest

from vector_store.faiss_store import build_index


def test_build_index_finds_identical_normalized_vector():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = build_index(embeddings)
    scores, indices = index.search(
        np.array([[1.0, 0.0]], dtype=np.float32),
        1,
    )

    assert isinstance(index, faiss.IndexFlatIP)
    assert index.ntotal == 2
    assert indices[0][0] == 0
    assert scores[0][0] == pytest.approx(1.0)


def test_build_index_rejects_empty_embeddings():
    embeddings = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="without embeddings"):
        build_index(embeddings)


def test_build_index_rejects_non_2d_embeddings():
    with pytest.raises(ValueError, match="2D array"):
        build_index(np.array([1.0, 2.0], dtype=np.float32))


def test_build_index_rejects_non_finite_embeddings():
    embeddings = np.array([[1.0, np.nan]], dtype=np.float32)

    with pytest.raises(ValueError, match="invalid numeric values"):
        build_index(embeddings)
