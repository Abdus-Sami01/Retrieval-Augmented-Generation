import logging
import pathlib
import tempfile

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from rag.cache import TTLCache
from rag.chunking import DocumentAwareChunker
from rag.config import Settings
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.generation.groq_client import GroqClient
from rag.observability import log_duration, setup_logging
from rag.pipeline import IngestPipeline, QueryPipeline
from rag.rate_limit import RateLimiter
from rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword_index import KeywordIndex
from rag.retrieval.query_rewriter import QueryRewriter
from rag.vectorstore.faiss_store import FaissVectorStore

logger = logging.getLogger("rag.api")


class QueryRequest(BaseModel):
    question: str
    source_path_filter: str | None = None
    tags_filter: list[str] | None = None


class CitationOut(BaseModel):
    source_path: str
    section_path: str | None
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sufficient_context: bool
    citations: list[CitationOut]


class IngestFileResult(BaseModel):
    filename: str
    ok: bool
    chunks_added: int
    error: str


class IngestResponse(BaseModel):
    results: list[IngestFileResult]


def build_app(
    settings: Settings | None = None,
    ingest_pipeline: IngestPipeline | None = None,
    query_pipeline: QueryPipeline | None = None,
    store: FaissVectorStore | None = None,
) -> FastAPI:
    """Assemble the FastAPI app. Pass ingest_pipeline/query_pipeline/store to inject
    fakes in tests and skip loading real embedding/LLM models."""
    settings = settings or Settings()
    setup_logging()

    if ingest_pipeline is None or query_pipeline is None or store is None:
        embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
        store = FaissVectorStore(dimension=embedder.dimension, data_dir=settings.data_dir)
        store.load()
        chunker = DocumentAwareChunker(
            chunk_size_tokens=settings.chunk_size_tokens, overlap_tokens=settings.chunk_overlap_tokens
        )
        llm = GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)

        keyword_index = None
        retriever = None
        if settings.use_hybrid_retrieval:
            keyword_index = KeywordIndex(data_dir=settings.data_dir)
            keyword_index.load()
            retriever = HybridRetriever(dense_store=store, keyword_index=keyword_index)

        reranker = CrossEncoderReranker(model_name=settings.reranker_model) if settings.use_reranker else None
        query_rewriter = QueryRewriter(llm=llm) if settings.use_query_rewriting else None
        query_cache = (
            TTLCache(ttl_seconds=settings.query_cache_ttl_seconds, max_size=settings.query_cache_max_size)
            if settings.query_cache_enabled
            else None
        )

        ingest_pipeline = IngestPipeline(
            chunker=chunker,
            embedder=embedder,
            store=store,
            keyword_index=keyword_index,
            redact_pii_on_ingest=settings.redact_pii_on_ingest,
        )
        query_pipeline = QueryPipeline(
            embedder=embedder,
            store=store,
            llm=llm,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
            retriever=retriever,
            reranker=reranker,
            query_rewriter=query_rewriter,
            rerank_candidate_k=settings.rerank_candidate_k,
            cache=query_cache,
            max_context_tokens=settings.max_context_tokens,
        )

    rate_limiter = RateLimiter(
        capacity=settings.rate_limit_requests_per_minute,
        refill_per_second=settings.rate_limit_requests_per_minute / 60.0,
    )

    def _enforce_rate_limit(request: Request) -> None:
        client_key = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")

    app = FastAPI(title="rag-platform")

    @app.get("/health")
    def health():
        return {"status": "ok", "indexed_chunks": store.count()}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: Request, files: list[UploadFile], tags: str = Form(default="")):
        _enforce_rate_limit(request)
        with log_duration(logger, "ingest", file_count=len(files)) as ctx:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] or None
            results = []
            for upload in files:
                suffix = pathlib.Path(upload.filename).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await upload.read())
                    tmp_path = tmp.name
                result = ingest_pipeline.ingest_file(tmp_path, display_name=upload.filename, tags=tag_list)
                results.append(
                    IngestFileResult(
                        filename=upload.filename,
                        ok=result.ok,
                        chunks_added=result.chunks_added,
                        error=result.error,
                    )
                )
            store.persist()
            ctx["chunks_added"] = sum(r.chunks_added for r in results)
            ctx["failures"] = sum(1 for r in results if not r.ok)
        return IngestResponse(results=results)

    @app.post("/query", response_model=QueryResponse)
    def query(http_request: Request, payload: QueryRequest):
        _enforce_rate_limit(http_request)
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        with log_duration(logger, "query", question_len=len(payload.question)) as ctx:
            filters = {}
            if payload.source_path_filter:
                filters["source_path"] = payload.source_path_filter
            if payload.tags_filter:
                filters["tags"] = payload.tags_filter
            answer = query_pipeline.answer(payload.question, filters=filters or None)
            ctx["sufficient_context"] = answer.sufficient_context
            ctx["citation_count"] = len(answer.citations)
        return QueryResponse(
            answer=answer.text,
            sufficient_context=answer.sufficient_context,
            citations=[
                CitationOut(
                    source_path=c.chunk.source_path,
                    section_path=c.chunk.section_path,
                    score=c.score,
                    text=c.chunk.text,
                )
                for c in answer.citations
            ],
        )

    return app
