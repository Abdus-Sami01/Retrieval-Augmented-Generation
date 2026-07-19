import tiktoken

from rag.models import RetrievedChunk

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def fit_to_budget(retrieved: list[RetrievedChunk], max_tokens: int) -> list[RetrievedChunk]:
    """Greedily keeps chunks in their given order (callers pass already relevance-sorted
    lists) until the next one would exceed max_tokens. Always keeps at least the first
    chunk even if it alone exceeds budget - generating from zero context is worse than
    one long chunk, and the LLM call will simply run a bit over budget in that edge case."""
    kept: list[RetrievedChunk] = []
    total = 0
    for retrieved_chunk in retrieved:
        chunk_tokens = count_tokens(retrieved_chunk.chunk.text)
        if kept and total + chunk_tokens > max_tokens:
            break
        kept.append(retrieved_chunk)
        total += chunk_tokens
    return kept
