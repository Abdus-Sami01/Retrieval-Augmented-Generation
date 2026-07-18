import os
import pathlib

import pytest

from rag.chunking import DocumentAwareChunker
from rag.config import Settings
from rag.embedding import SentenceTransformerEmbedder
from rag.generation.groq_client import GroqClient
from rag.pipeline import IngestPipeline, QueryPipeline
from rag.vectorstore import FaissVectorStore

CORPUS_DIR = pathlib.Path(__file__).parent.parent / "corpus_sample"
_TEST_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def real_embedder():
    return SentenceTransformerEmbedder(model_name=_TEST_EMBED_MODEL)


def test_e2e_ingest_and_retrieve_real_embeddings(tmp_path, real_embedder):
    store = FaissVectorStore(dimension=real_embedder.dimension, data_dir=str(tmp_path))
    ingest = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=300, overlap_tokens=40),
        embedder=real_embedder,
        store=store,
    )
    paths = [str(p) for p in CORPUS_DIR.glob("*.md")]
    assert paths, "sample corpus must contain at least one document"

    results = ingest.ingest_files(paths)
    assert all(r.ok for r in results), [r.error for r in results if not r.ok]
    assert store.count() > 0

    query_vector = real_embedder.embed_query(
        "What does the faithfulness harness check about generated answers?"
    )
    retrieved = store.search(query_vector, top_k=3)
    assert retrieved
    top_sources = [r.chunk.source_path for r in retrieved]
    assert any("rag-faithfulness-harness" in s for s in top_sources)


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_GROQ_TESTS"),
    reason="set RUN_LIVE_GROQ_TESTS=1 to run a real Groq API call",
)
def test_e2e_full_pipeline_live_groq(tmp_path, real_embedder):
    settings = Settings()
    store = FaissVectorStore(dimension=real_embedder.dimension, data_dir=str(tmp_path))
    ingest = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=300, overlap_tokens=40),
        embedder=real_embedder,
        store=store,
    )
    ingest.ingest_files([str(p) for p in CORPUS_DIR.glob("*.md")])

    llm = GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)
    query = QueryPipeline(embedder=real_embedder, store=store, llm=llm, min_score=0.2)
    answer = query.answer("What does the retrieval-verification-gate project do?")
    assert answer.sufficient_context
    assert answer.citations
