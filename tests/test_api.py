from fastapi.testclient import TestClient

from rag.api import build_app
from rag.chunking import DocumentAwareChunker
from rag.config import Settings
from rag.pipeline import IngestPipeline, QueryPipeline
from tests.test_pipeline import EmptyStore, FakeEmbedder, FakeLLM, FakeStore


def _client(store=None, llm_response="grounded answer [1]."):
    store = store or FakeStore()
    settings = Settings(_env_file=None, groq_api_key="test-key")
    ingest_pipeline = IngestPipeline(
        chunker=DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60),
        embedder=FakeEmbedder(),
        store=store,
    )
    query_pipeline = QueryPipeline(
        embedder=FakeEmbedder(), store=store, llm=FakeLLM(llm_response), min_score=0.35
    )
    app = build_app(
        settings=settings, ingest_pipeline=ingest_pipeline, query_pipeline=query_pipeline, store=store
    )
    return TestClient(app), store


def test_health_endpoint():
    client, _ = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_endpoint_accepts_file(tmp_path):
    client, store = _client()
    files = [("files", ("note.txt", b"hello world, this is a real test document.", "text/plain"))]
    resp = client.post("/ingest", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["ok"] is True
    assert store.count() > 0


def test_query_endpoint_rejects_empty_question():
    client, _ = _client()
    resp = client.post("/query", json={"question": "   "})
    assert resp.status_code == 400


def test_query_endpoint_returns_insufficient_context():
    client, _ = _client(store=EmptyStore())
    resp = client.post("/query", json={"question": "what is x?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sufficient_context"] is False
    assert body["citations"] == []


def test_query_endpoint_returns_grounded_answer():
    client, store = _client()
    client.post(
        "/ingest",
        files=[("files", ("note.txt", b"widgets are small testing devices.", "text/plain"))],
    )
    resp = client.post("/query", json={"question": "what are widgets?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sufficient_context"] is True
    assert len(body["citations"]) >= 1
