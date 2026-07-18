import pathlib

from pypdf import PdfReader

from rag.loaders.base import DocumentLoader
from rag.models import Document


class PdfLoader(DocumentLoader):
    def load(self, path: str) -> Document:
        file_path = pathlib.Path(path)
        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            raise ValueError(f"unreadable pdf: {path}") from exc

        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        text = "\n\n".join(pages).strip()
        if not text:
            raise ValueError(f"no extractable text in pdf: {path}")
        return Document(text=text, source_path=str(file_path), doc_type="pdf")
