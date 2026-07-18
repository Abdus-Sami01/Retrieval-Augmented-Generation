import hashlib
from dataclasses import dataclass, field


def _stable_doc_id(source_path: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_path}:{len(text)}".encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class Document:
    text: str
    source_path: str
    doc_type: str
    doc_id: str = field(default="")

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = _stable_doc_id(self.source_path, self.text)


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    source_path: str
    section_path: str | None
    char_start: int
    char_end: int

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::{self.chunk_index}"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    citations: list[RetrievedChunk]
    sufficient_context: bool
