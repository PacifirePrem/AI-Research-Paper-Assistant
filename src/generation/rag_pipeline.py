import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)
model.eval()


def generate_answer(query, context_chunks):
    excerpts = []

    for chunk in context_chunks:
        excerpts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(excerpts)

    if not context.strip():
        return "I couldn't find relevant information in the paper."

    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about research papers. "
                "Use only the supplied excerpts. "
                "Do not invent facts, figures, results, or dataset details. "
                "If the excerpts do not support an answer, say so. "
                "Cite supporting pages using the format [Page N]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research paper excerpts:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer clearly and cite the pages supporting your answer."
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    answer_tokens = output[0][input_length:]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer or "I couldn't generate an answer from the retrieved excerpts."
