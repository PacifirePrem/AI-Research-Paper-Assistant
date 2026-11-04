from pypdf import PdfReader


def load_pdf(file_path):
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        list[dict]: Each item contains page number and extracted text.
    """
    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    if not pages:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "It may be scanned or image-only."
        )

    return pages


if __name__ == "__main__":
    file_path = "data/papers/sample_paper.pdf"
    pages = load_pdf(file_path)

    print(f"Extracted text from {len(pages)} pages.")

    for page in pages[:2]:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:500])
