import pytest

from ingestion import pdf_loader


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, _file_path):
        self.pages = [
            FakePage("  First page text  "),
            FakePage("   "),
            FakePage(None),
            FakePage("Fourth page text"),
        ]


def test_load_pdf_preserves_original_page_numbers(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"fake PDF content")

    monkeypatch.setattr(pdf_loader, "PdfReader", FakeReader)

    pages = pdf_loader.load_pdf(pdf_path)

    assert pages == [
        {"page": 1, "text": "First page text"},
        {"page": 4, "text": "Fourth page text"},
    ]


def test_load_pdf_rejects_documents_without_extractable_text(
    tmp_path, monkeypatch
):
    class EmptyReader:
        def __init__(self, _file_path):
            self.pages = [FakePage("   "), FakePage(None)]

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"fake PDF content")

    monkeypatch.setattr(pdf_loader, "PdfReader", EmptyReader)

    with pytest.raises(ValueError, match="No extractable text"):
        pdf_loader.load_pdf(pdf_path)
