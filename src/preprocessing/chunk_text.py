def chunk_text(text, chunk_size=250, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    sample_text = "This is a sample research paper text. " * 200

    chunks = chunk_text(sample_text)

    print("Number of chunks:", len(chunks))

    print("\nFirst chunk:\n")

    print(chunks[0])