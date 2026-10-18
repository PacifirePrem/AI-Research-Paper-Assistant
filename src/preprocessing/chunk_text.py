def chunk_text(pages, chunk_size=1000, overlap=200):
    """
    Create overlapping chunks while preserving page metadata.

    Args:
        pages: List of {"page": page_number, "text": page_text} records.

    Returns:
        List of {"page": page_number, "text": chunk_text} records.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    chunks = []

    for page_record in pages:
        page_number = page_record["page"]
        text = page_record["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer ending at a nearby sentence or word boundary.
            if end < len(text):
                boundary = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind(" ", start, end),
                )

                if boundary > start + chunk_size // 2:
                    end = boundary + 1

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "page": page_number,
                    "text": chunk,
                })

            if end >= len(text):
                break

            start = max(start + 1, end - overlap)

    return chunks
