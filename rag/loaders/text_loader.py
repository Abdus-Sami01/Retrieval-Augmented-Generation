import pathlib

from rag.loaders.base import DocumentLoader
from rag.models import Document


class TextLoader(DocumentLoader):
    def load(self, path: str) -> Document:
        file_path = pathlib.Path(path)
        text = file_path.read_text(encoding="utf-8", errors="strict")
        if not text.strip():
            raise ValueError(f"empty file: {path}")
        doc_type = "markdown" if file_path.suffix.lower() == ".md" else "text"
        return Document(text=text, source_path=str(file_path), doc_type=doc_type)
