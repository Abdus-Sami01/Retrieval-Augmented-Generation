from dataclasses import dataclass

from rag.chunking import DocumentAwareChunker
from rag.embedding.base import Embedder
from rag.generation.base import LLMClient
from rag.generation.prompt import NO_CONTEXT_MARKER, build_prompt
from rag.loaders import load_path
from rag.models import Answer
from rag.vectorstore.base import VectorStore

INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough information in the indexed documents to answer that."


@dataclass
class IngestResult:
    path: str
    ok: bool
    chunks_added: int = 0
    error: str = ""


class IngestPipeline:
    def __init__(self, chunker: DocumentAwareChunker, embedder: Embedder, store: VectorStore):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def ingest_file(
        self, path: str, display_name: str | None = None, tags: list[str] | None = None
    ) -> IngestResult:
        try:
            document = load_path(path)
        except Exception as exc:
            return IngestResult(path=path, ok=False, error=str(exc))

        if display_name:
            document.source_path = display_name

        chunks = self.chunker.chunk(document)
        if not chunks:
            return IngestResult(path=path, ok=False, error="no chunks produced")

        if tags:
            for chunk in chunks:
                chunk.tags = list(tags)

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return IngestResult(path=path, ok=True, chunks_added=len(chunks))

    def ingest_files(self, paths: list[str], tags: list[str] | None = None) -> list[IngestResult]:
        results = [self.ingest_file(p, tags=tags) for p in paths]
        self.store.persist()
        return results


class QueryPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        top_k: int = 5,
        min_score: float = 0.35,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def answer(self, question: str, filters: dict | None = None) -> Answer:
        query_vector = self.embedder.embed_query(question)
        retrieved = self.store.search(query_vector, top_k=self.top_k, filters=filters)
        sufficient = [r for r in retrieved if r.score >= self.min_score]

        if not sufficient:
            return Answer(text=INSUFFICIENT_CONTEXT_MESSAGE, citations=[], sufficient_context=False)

        system_prompt, user_prompt = build_prompt(question, sufficient)
        raw_text = self.llm.complete(system_prompt, user_prompt)

        if NO_CONTEXT_MARKER in raw_text:
            return Answer(text=INSUFFICIENT_CONTEXT_MESSAGE, citations=[], sufficient_context=False)

        return Answer(text=raw_text, citations=sufficient, sufficient_context=True)
