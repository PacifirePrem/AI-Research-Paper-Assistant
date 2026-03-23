import ollama


def generate_answer(query, context_chunks):

    # Take top relevant chunks
    context = "\n".join(context_chunks[:3])

    prompt = f"""
You are an AI research assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer clearly and professionally:
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"].strip()