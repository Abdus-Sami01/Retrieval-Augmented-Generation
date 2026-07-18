import os

from rag.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "250")
    settings = Settings()
    assert settings.groq_api_key == "test-key-123"
    assert settings.chunk_size_tokens == 250


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    settings = Settings()
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.embedding_model == "BAAI/bge-base-en-v1.5"
    assert settings.retrieval_top_k == 5
    assert 0.0 <= settings.retrieval_min_score <= 1.0


def test_settings_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    try:
        Settings(_env_file=None)
        assert False, "expected missing GROQ_API_KEY to raise"
    except Exception:
        pass
