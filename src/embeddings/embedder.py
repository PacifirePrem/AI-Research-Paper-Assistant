from sentence_transformers import SentenceTransformer

# Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings


if __name__ == "__main__":

    chunks = [
        "Transformers are a neural network architecture.",
        "Attention mechanisms allow models to focus on important words."
    ]

    embeddings = create_embeddings(chunks)

    print("Number of embeddings:", len(embeddings))
    print("Embedding vector size:", len(embeddings[0]))