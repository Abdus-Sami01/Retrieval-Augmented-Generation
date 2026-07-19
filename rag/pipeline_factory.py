from rag.cache import TTLCache
from rag.chunking import DocumentAwareChunker
from rag.config import Settings
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.generation.groq_client import GroqClient
from rag.pipeline import IngestPipeline, QueryPipeline
from rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword_index import KeywordIndex
from rag.retrieval.query_rewriter import QueryRewriter
from rag.vectorstore.faiss_store import FaissVectorStore


def build_pipelines(settings: Settings) -> tuple[IngestPipeline, QueryPipeline, FaissVectorStore]:
    """Assembles real (non-fake) ingest/query pipelines from Settings. Shared by the
    FastAPI app (rag/api.py) and the eval CLI (rag/evaluation/cli.py) so both build
    identical pipelines from the same config instead of duplicating this wiring."""
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
    return ingest_pipeline, query_pipeline, store
