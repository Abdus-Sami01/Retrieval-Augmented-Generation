from rag.models import RetrievedChunk

NO_CONTEXT_MARKER = "NOT_FOUND_IN_CONTEXT"

SYSTEM_PROMPT = f"""You are a precise research assistant. Answer ONLY using the numbered \
context passages provided below. Rules:
1. Every factual claim in your answer must be traceable to a passage. Cite passages inline \
using the format [n] where n is the passage number.
2. If the passages do not contain enough information to answer the question, respond with \
exactly: {NO_CONTEXT_MARKER}
3. Never use outside knowledge, never guess, never pad with generic statements not grounded \
in the passages.
"""


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    if not retrieved:
        passages_block = "(no passages retrieved)"
    else:
        lines = []
        for i, r in enumerate(retrieved, start=1):
            lines.append(f"[{i}] (source: {r.chunk.source_path}) {r.chunk.text}")
        passages_block = "\n\n".join(lines)

    user_prompt = (
        f"Context passages:\n{passages_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above, citing [n] for each claim."
    )
    return SYSTEM_PROMPT, user_prompt
