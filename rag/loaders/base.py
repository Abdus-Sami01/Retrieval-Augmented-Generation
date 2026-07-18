from abc import ABC, abstractmethod

from rag.models import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Document:
        """Load a single file into a Document. Raises on unreadable/corrupt files."""
        raise NotImplementedError
