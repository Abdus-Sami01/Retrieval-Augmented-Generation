from abc import ABC, abstractmethod

from rag.models import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Re-score and re-sort candidates by query relevance, returning at most top_k."""
        raise NotImplementedError
