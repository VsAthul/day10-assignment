import ollama


def build_prompt(user_question: str, context_chunks: list[str]) -> str:
    """Construct a prompt using retrieved context chunks and the user's question."""
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context available."
    prompt = (
        f"You are a helpful assistant. Answer the user's question ONLY based on the "
        f"context provided below. If the answer is not found in the context, say "
        f"'I could not find relevant information in the document.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n\n"
        f"Answer:"
    )
    return prompt


def generate_ai_response(
    user_question: str,
    context_chunks: list[str],
    model: str = "qwen3:0.6b",
) -> str:
    """Build a prompt from context and generate an answer using Ollama."""
    prompt = build_prompt(user_question, context_chunks)

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable banking document assistant. "
                    "Answer questions strictly based on the provided context. "
                    "Be concise, accurate, and professional."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"].strip()
