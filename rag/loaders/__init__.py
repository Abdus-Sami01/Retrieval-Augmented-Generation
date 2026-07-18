from rag.loaders.base import DocumentLoader
from rag.loaders.pdf_loader import PdfLoader
from rag.loaders.text_loader import TextLoader

__all__ = ["DocumentLoader", "TextLoader", "PdfLoader", "load_path"]

_LOADERS_BY_SUFFIX = {
    ".txt": TextLoader(),
    ".md": TextLoader(),
    ".pdf": PdfLoader(),
}


def load_path(path: str):
    import pathlib

    suffix = pathlib.Path(path).suffix.lower()
    loader = _LOADERS_BY_SUFFIX.get(suffix)
    if loader is None:
        raise ValueError(f"no loader registered for file type '{suffix}'")
    return loader.load(path)
