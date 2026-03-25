from transformers import pipeline

# Use lightweight model for deployment
generator = pipeline("text-generation", model="distilgpt2")


def generate_answer(query, context_chunks):

    context = " ".join(context_chunks[:2])

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    output = generator(
        prompt,
        max_length=120,
        num_return_sequences=1,
        do_sample=False
    )

    generated_text = output[0]["generated_text"]

    # Remove prompt from output
    answer = generated_text.replace(prompt, "").strip()

    return answer