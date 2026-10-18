import pytest

from preprocessing.chunk_text import chunk_text


def test_chunk_text_preserves_page_numbers():
    pages = [
        {"page": 2, "text": "Alpha beta gamma delta epsilon zeta eta theta."},
        {"page": 5, "text": "Second page contains additional words for testing."},
    ]

    chunks = chunk_text(pages, chunk_size=15, overlap=3)

    assert chunks
    assert {chunk["page"] for chunk in chunks} == {2, 5}
    assert all(chunk["text"].strip() for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (10, -1),
        (10, 10),
    ],
)
def test_chunk_text_rejects_invalid_settings(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text(
            [{"page": 1, "text": "Sample text"}],
            chunk_size=chunk_size,
            overlap=overlap,
        )
