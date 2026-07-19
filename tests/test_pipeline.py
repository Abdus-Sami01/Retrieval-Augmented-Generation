from rag.cache import TTLCache
from rag.chunking import DocumentAwareChunker
from rag.embedding.base import Embedder
from rag.generation.base import LLMClient
from rag.generation.prompt import NO_CONTEXT_MARKER
from rag.models import Chunk, RetrievedChunk
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

    def search(self, query_vector, top_k, filters=None):
        candidates = self.chunks
        if filters:
            wanted_source = filters.get("source_path")
            wanted_tags = filters.get("tags")
            candidates = [
                c
                for c in candidates
                if (not wanted_source or wanted_source in c.source_path)
                and (not wanted_tags or set(wanted_tags) & set(c.tags))
            ]
        return [RetrievedChunk(chunk=c, score=0.9) for c in candidates[:top_k]]

    def count(self):
        return len(self.chunks)

    def persist(self):
        self.persisted = True

    def load(self):
        pass


class EmptyStore(FakeStore):
    def search(self, query_vector, top_k, filters=None):
        return []


class FakeLLM(LLMClient):
    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    def complete(self, system_prompt, user_prompt):
        self.call_count += 1
        return self._response


class FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.last_call = None

    def search(self, query_text, query_vector, top_k, filters=None):
        self.last_call = {"query_text": query_text, "top_k": top_k, "filters": filters}
        return self._results[:top_k]


class FakeReranker:
    def __init__(self, rescored):
        self._rescored = rescored
        self.last_call = None

    def rerank(self, query, candidates, top_k):
        self.last_call = {"query": query, "candidates": candidates, "top_k": top_k}
        return self._rescored[:top_k]


class FakeQueryRewriter:
    def __init__(self, rewritten: str):
        self._rewritten = rewritten

    def rewrite(self, query):
        return self._rewritten


class FakeKeywordIndex:
    def __init__(self):
        self.added_chunks = []
        self.persisted = False

    def add(self, chunks):
        self.added_chunks.extend(chunks)

    def persist(self):
        self.persisted = True


def test_ingest_file_adds_to_keyword_index_when_present(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world, this is a test document with enough content.", encoding="utf-8")
    store = FakeStore()
    keyword_index = FakeKeywordIndex()
    pipeline = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
        keyword_index=keyword_index,
    )
    pipeline.ingest_files([str(f)])
    assert len(keyword_index.added_chunks) == 1
    assert keyword_index.persisted


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


def test_ingest_file_attaches_tags_to_every_chunk(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world, this is a test document with enough content.", encoding="utf-8")
    store = FakeStore()
    pipeline = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    pipeline.ingest_file(str(f), tags=["legal", "2026"])
    assert store.chunks[0].tags == ["legal", "2026"]


def test_query_answer_passes_filters_through_to_store(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("widgets are small mechanical devices used in testing.", encoding="utf-8")
    store = FakeStore()
    ingest = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    ingest.ingest_file(str(f))

    received_filters = {}

    original_search = store.search

    def spying_search(query_vector, top_k, filters=None):
        received_filters["value"] = filters
        return original_search(query_vector, top_k, filters)

    store.search = spying_search

    query = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM("Widgets are devices [1]."), min_score=0.35)
    query.answer("what are widgets?", filters={"tags": ["legal"]})
    assert received_filters["value"] == {"tags": ["legal"]}


def test_ingest_file_uses_display_name_for_citations(tmp_path):
    # Regression: /ingest saves uploads to a temp path before processing; without a
    # display_name override, citations would point at a throwaway temp filename
    # instead of the name the user actually uploaded.
    f = tmp_path / "tmpXYZ123.txt"
    f.write_text("original content the user uploaded as report.txt", encoding="utf-8")
    store = FakeStore()
    pipeline = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    pipeline.ingest_file(str(f), display_name="report.txt")
    assert store.chunks[0].source_path == "report.txt"


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


def _widget_chunk(tags=None):
    return Chunk(
        text="widgets are small mechanical devices",
        doc_id="d1",
        chunk_index=0,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=10,
        tags=tags or [],
    )


def test_query_uses_retriever_when_provided():
    retriever = FakeRetriever([RetrievedChunk(chunk=_widget_chunk(), score=0.9)])
    query = QueryPipeline(
        embedder=FakeEmbedder(),
        store=FakeStore(),
        llm=FakeLLM("Widgets are devices [1]."),
        min_score=0.35,
        retriever=retriever,
    )
    answer = query.answer("what are widgets?", filters={"tags": ["legal"]})
    assert answer.sufficient_context
    assert retriever.last_call["filters"] == {"tags": ["legal"]}


def test_query_with_reranker_ignores_min_score_on_reranker_output():
    # Regression: cross-encoder sigmoid scores are not on cosine's 0..1 scale in any
    # reliable way (measured live: ~0.02 for a genuinely correct top match), so
    # min_score must not be applied to reranker output - only to the plain-cosine path.
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    reranker = FakeReranker([RetrievedChunk(chunk=_widget_chunk(), score=0.05)])
    query = QueryPipeline(
        embedder=FakeEmbedder(),
        store=store,
        llm=FakeLLM("Widgets are devices [1]."),
        min_score=0.35,
        reranker=reranker,
    )
    answer = query.answer("what are widgets?")
    assert answer.sufficient_context


def test_query_with_reranker_returning_empty_list_is_insufficient():
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    reranker = FakeReranker([])
    query = QueryPipeline(
        embedder=FakeEmbedder(),
        store=store,
        llm=FakeLLM("Widgets are devices [1]."),
        min_score=0.35,
        reranker=reranker,
    )
    answer = query.answer("what are widgets?")
    assert not answer.sufficient_context


def test_query_rewriter_used_for_retrieval_not_final_prompt():
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    rewriter = FakeQueryRewriter("what are small mechanical devices called?")
    query = QueryPipeline(
        embedder=FakeEmbedder(),
        store=store,
        llm=FakeLLM("Widgets are devices [1]."),
        min_score=0.35,
        query_rewriter=rewriter,
    )
    answer = query.answer("wut r widgets")
    assert answer.sufficient_context


def test_query_without_reranker_or_retriever_behaves_as_before():
    # Backward-compat guard: no new kwargs set should reproduce the original,
    # plain store.search()-only code path exactly.
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    query = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM("Widgets are devices [1]."))
    answer = query.answer("what are widgets?")
    assert answer.sufficient_context
    assert len(answer.citations) == 1


def test_query_cache_hit_skips_llm_call():
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    llm = FakeLLM("Widgets are devices [1].")
    query = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=llm, cache=TTLCache(ttl_seconds=60))

    first = query.answer("what are widgets?")
    second = query.answer("what are widgets?")

    assert llm.call_count == 1
    assert second.text == first.text


def test_query_cache_miss_for_different_filters():
    store = FakeStore()
    store.chunks.append(_widget_chunk(tags=["legal"]))
    llm = FakeLLM("Widgets are devices [1].")
    query = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=llm, cache=TTLCache(ttl_seconds=60))

    query.answer("what are widgets?", filters=None)
    query.answer("what are widgets?", filters={"tags": ["legal"]})

    assert llm.call_count == 2


def test_query_without_cache_calls_llm_every_time():
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    llm = FakeLLM("Widgets are devices [1].")
    query = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=llm)

    query.answer("what are widgets?")
    query.answer("what are widgets?")

    assert llm.call_count == 2
