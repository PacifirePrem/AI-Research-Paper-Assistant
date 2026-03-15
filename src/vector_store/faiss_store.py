import faiss
import numpy as np


def build_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


if __name__ == "__main__":

    embeddings = np.random.rand(10, 384).astype("float32")

    index = build_index(embeddings)

    print("Number of vectors in index:", index.ntotal)