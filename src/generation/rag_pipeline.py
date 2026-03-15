from transformers import pipeline

# Load text generation model
generator = pipeline("text-generation", model="gpt2")


def generate_answer(query, context_chunks):

    context = " ".join(context_chunks)

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    output = generator(prompt, max_length=200, num_return_sequences=1)

    return output[0]["generated_text"]


if __name__ == "__main__":

    query = "What is attention in transformers?"

    context_chunks = [
        "Transformers use attention mechanisms to process sequences.",
        "Self-attention helps models focus on relevant tokens."
    ]

    answer = generate_answer(query, context_chunks)

    print("\nGenerated Answer:\n")
    print(answer)