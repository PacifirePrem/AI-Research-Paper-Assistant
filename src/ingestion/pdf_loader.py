from pypdf import PdfReader


def load_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


if __name__ == "__main__":

    file_path = "data/papers/sample_paper.pdf"

    text = load_pdf(file_path)

    print("First 500 characters of paper:\n")
    print(text[:500])