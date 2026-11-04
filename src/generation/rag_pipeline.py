
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
                "You are an assistant that answers questions about "
                "research papers.\n"
                "Use only the supplied research paper excerpts.\n"
                "Do not invent facts, figures, results, or dataset details.\n"
                "If the excerpts do not support an answer, say so.\n"
                "Start every answer with a complete, grammatically correct "
                "sentence.\n"
                "Never start with a partial word, incomplete sentence, "
                "ellipsis, or continuation fragment.\n"
                "Paraphrase incomplete source passages into complete "
                "sentences.\n"
                "Cite supporting pages using [Page N]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research paper excerpts:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Write a clear, self-contained answer. "
                "Begin with the main point in a complete sentence. "
                "Include page citations where supported."
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

    # Store the prompt length BEFORE generating new tokens.
    prompt_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract only newly generated answer tokens.
    answer_tokens = output[0, prompt_length:]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if not answer:
        return "I couldn't generate an answer from the retrieved excerpts."

    return answer
