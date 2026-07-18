from rag.generation.base import LLMClient

_SYSTEM_PROMPT = """You rewrite short or ambiguous search queries into clearer, more specific \
standalone queries for document retrieval. Preserve the original meaning and intent exactly \
- do not introduce new facts or assumptions. If the query is already clear and specific, \
return it completely unchanged. Return ONLY the rewritten query text, nothing else - no \
preamble, no quotes, no explanation."""


class QueryRewriter:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        try:
            rewritten = self.llm.complete(_SYSTEM_PROMPT, f"Query: {query}").strip()
        except Exception:
            # Rewriting is a retrieval-quality enhancement, not a hard dependency -
            # a rewrite failure should never block the underlying query from running.
            return query
        return rewritten if rewritten else query
