from rag.chunking import DocumentAwareChunker
from rag.embedding.base import Embedder
from rag.generation.base import LLMClient
from rag.generation.prompt import NO_CONTEXT_MARKER
from rag.models import RetrievedChunk
from rag.pipeline import INSUFFICIENT_CONTEXT_MESSAGE, IngestPipeline, QueryPipeline
from rag.vectorstore.base import VectorStore


class FakeEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class FakeStore(VectorStore):
    def __init__(self):
        self.chunks = []
        self.vectors = []
        self.persisted = False

    def add(self, chunks, vectors):
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def search(self, query_vector, top_k):
        return [RetrievedChunk(chunk=c, score=0.9) for c in self.chunks[:top_k]]

    def count(self):
        return len(self.chunks)

    def persist(self):
        self.persisted = True

    def load(self):
        pass


class EmptyStore(FakeStore):
    def search(self, query_vector, top_k):
        return []


class FakeLLM(LLMClient):
    def __init__(self, response: str):
        self._response = response

    def complete(self, system_prompt, user_prompt):
        return self._response


def test_ingest_file_success(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world, this is a test document with enough content.", encoding="utf-8")
    store = FakeStore()
    pipeline = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    result = pipeline.ingest_file(str(f))
    assert result.ok
    assert result.chunks_added == 1
    assert store.count() == 1


def test_ingest_file_failure_on_missing_file(tmp_path):
    store = FakeStore()
    pipeline = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    result = pipeline.ingest_file(str(tmp_path / "missing.txt"))
    assert not result.ok
    assert result.error


def test_ingest_files_persists_store(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("some real content for ingestion testing purposes.", encoding="utf-8")
    store = FakeStore()
    pipeline = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    pipeline.ingest_files([str(f)])
    assert store.persisted


def test_query_returns_insufficient_when_no_retrieval():
    pipeline = QueryPipeline(embedder=FakeEmbedder(), store=EmptyStore(), llm=FakeLLM("anything"), min_score=0.35)
    answer = pipeline.answer("what is x?")
    assert not answer.sufficient_context
    assert answer.text == INSUFFICIENT_CONTEXT_MESSAGE
    assert answer.citations == []


def test_query_returns_grounded_answer_with_citations(tmp_path):
    store = FakeStore()
    f = tmp_path / "a.txt"
    f.write_text("widgets are small mechanical devices used in testing.", encoding="utf-8")
    ingest = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    ingest.ingest_file(str(f))

    query = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM("Widgets are devices [1]."), min_score=0.35)
    answer = query.answer("what are widgets?")
    assert answer.sufficient_context
    assert len(answer.citations) == 1
    assert "[1]" in answer.text


def test_query_returns_insufficient_when_llm_signals_no_context(tmp_path):
    store = FakeStore()
    f = tmp_path / "a.txt"
    f.write_text("some unrelated content about gardening tips.", encoding="utf-8")
    ingest = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    ingest.ingest_file(str(f))

    query = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM(NO_CONTEXT_MARKER), min_score=0.35)
    answer = query.answer("what is quantum computing?")
    assert not answer.sufficient_context
    assert answer.text == INSUFFICIENT_CONTEXT_MESSAGE
