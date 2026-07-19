from fastapi.testclient import TestClient

from rag.api import build_app
from rag.chunking import DocumentAwareChunker
from rag.config import Settings
from rag.pipeline import IngestPipeline, QueryPipeline
from tests.test_pipeline import EmptyStore, FakeEmbedder, FakeLLM, FakeStore


def _client(store=None, llm_response="grounded answer [1].", rate_limit_requests_per_minute=60):
    store = store or FakeStore()
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        rate_limit_requests_per_minute=rate_limit_requests_per_minute,
    )
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


def test_query_endpoint_returns_429_after_rate_limit_exceeded():
    client, _ = _client(rate_limit_requests_per_minute=1)
    first = client.post("/query", json={"question": "what is x?"})
    second = client.post("/query", json={"question": "what is x?"})
    assert first.status_code != 429
    assert second.status_code == 429


def test_ingest_endpoint_returns_429_after_rate_limit_exceeded():
    client, _ = _client(rate_limit_requests_per_minute=1)
    files = [("files", ("note.txt", b"hello world, this is a real test document.", "text/plain"))]
    first = client.post("/ingest", files=files)
    second = client.post("/ingest", files=files)
    assert first.status_code != 429
    assert second.status_code == 429


def test_query_endpoint_logs_structured_event(caplog):
    import logging

    client, store = _client()
    client.post(
        "/ingest",
        files=[("files", ("note.txt", b"widgets are small testing devices.", "text/plain"))],
    )
    with caplog.at_level(logging.INFO, logger="rag.api"):
        client.post("/query", json={"question": "what are widgets?"})

    query_records = [r for r in caplog.records if getattr(r, "event", None) == "query"]
    assert len(query_records) == 1
    assert query_records[0].status == "ok"
    assert query_records[0].duration_ms >= 0
    assert query_records[0].sufficient_context is True


def test_ingest_endpoint_attaches_tags():
    client, store = _client()
    client.post(
        "/ingest",
        files=[("files", ("note.txt", b"widgets are small testing devices.", "text/plain"))],
        data={"tags": "legal, 2026"},
    )
    assert store.chunks[0].tags == ["legal", "2026"]


def test_query_endpoint_applies_tags_filter():
    client, store = _client()
    client.post(
        "/ingest",
        files=[("files", ("note.txt", b"widgets are small testing devices.", "text/plain"))],
        data={"tags": "engineering"},
    )
    resp = client.post("/query", json={"question": "what are widgets?", "tags_filter": ["legal"]})
    body = resp.json()
    assert body["sufficient_context"] is False
    assert body["citations"] == []
