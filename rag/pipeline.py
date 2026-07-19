import hashlib
import json
from dataclasses import dataclass

from rag.cache import TTLCache
from rag.chunking import DocumentAwareChunker
from rag.embedding.base import Embedder
from rag.generation.base import LLMClient
from rag.generation.prompt import NO_CONTEXT_MARKER, build_prompt
from rag.loaders import load_path
from rag.models import Answer
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword_index import KeywordIndex
from rag.retrieval.query_rewriter import QueryRewriter
from rag.retrieval.reranker_base import Reranker
from rag.vectorstore.base import VectorStore

INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough information in the indexed documents to answer that."


def _cache_key(question: str, filters: dict | None) -> str:
    payload = json.dumps({"question": question, "filters": filters or {}}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class IngestResult:
    path: str
    ok: bool
    chunks_added: int = 0
    error: str = ""


class IngestPipeline:
    def __init__(
        self,
        chunker: DocumentAwareChunker,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex | None = None,
    ):
        """keyword_index is optional: set it when hybrid retrieval is enabled so every
        ingested chunk lands in both the dense store and the BM25 index."""
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.keyword_index = keyword_index

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
        if self.keyword_index is not None:
            self.keyword_index.add(chunks)
        return IngestResult(path=path, ok=True, chunks_added=len(chunks))

    def ingest_files(self, paths: list[str], tags: list[str] | None = None) -> list[IngestResult]:
        results = [self.ingest_file(p, tags=tags) for p in paths]
        self.store.persist()
        if self.keyword_index is not None:
            self.keyword_index.persist()
        return results


class QueryPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        top_k: int = 5,
        min_score: float = 0.35,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        query_rewriter: QueryRewriter | None = None,
        rerank_candidate_k: int = 20,
        cache: TTLCache | None = None,
    ):
        """retriever, reranker, and query_rewriter are all optional, additive stages:
        with none set this behaves exactly like the original store.search()-only
        pipeline. retriever (HybridRetriever) replaces plain dense search when set.

        reranker re-scores candidates with a cross-encoder. Its raw logits (squashed
        through a sigmoid) are NOT comparable to cosine similarity's 0..1 scale in any
        reliable, corpus-independent way - measured live, this model's sigmoid score
        for a genuinely correct top match was ~0.02, far under min_score's 0.35
        default. So when a reranker is set, min_score is not applied to its output;
        sufficiency instead relies on the reranker's ranking being non-empty plus the
        LLM's own NO_CONTEXT_MARKER self-check in the generation step. min_score still
        gates the plain-cosine, no-reranker path exactly as before."""
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.retriever = retriever
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.rerank_candidate_k = rerank_candidate_k
        self.cache = cache

    def answer(self, question: str, filters: dict | None = None) -> Answer:
        if self.cache is not None:
            cache_key = _cache_key(question, filters)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        computed = self._answer_uncached(question, filters)

        if self.cache is not None:
            self.cache.set(cache_key, computed)
        return computed

    def _answer_uncached(self, question: str, filters: dict | None) -> Answer:
        retrieval_query = self.query_rewriter.rewrite(question) if self.query_rewriter else question
        query_vector = self.embedder.embed_query(retrieval_query)

        fetch_k = self.rerank_candidate_k if self.reranker else self.top_k
        if self.retriever is not None:
            retrieved = self.retriever.search(retrieval_query, query_vector, top_k=fetch_k, filters=filters)
        else:
            retrieved = self.store.search(query_vector, top_k=fetch_k, filters=filters)

        if self.reranker is not None:
            retrieved = self.reranker.rerank(retrieval_query, retrieved, top_k=self.top_k)
            sufficient = retrieved
        else:
            sufficient = [r for r in retrieved if r.score >= self.min_score]

        if not sufficient:
            return Answer(text=INSUFFICIENT_CONTEXT_MESSAGE, citations=[], sufficient_context=False)

        system_prompt, user_prompt = build_prompt(question, sufficient)
        raw_text = self.llm.complete(system_prompt, user_prompt)

        if NO_CONTEXT_MARKER in raw_text:
            return Answer(text=INSUFFICIENT_CONTEXT_MESSAGE, citations=[], sufficient_context=False)

        return Answer(text=raw_text, citations=sufficient, sufficient_context=True)
