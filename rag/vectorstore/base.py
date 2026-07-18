from abc import ABC, abstractmethod

from rag.models import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        """filters recognized keys: 'source_path' (substring match), 'tags' (any-of match)."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def persist(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError
