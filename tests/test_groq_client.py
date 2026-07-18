import httpx
import pytest
from groq import APIStatusError

from rag.generation.groq_client import GroqClient


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=status_code, request=request)
    return APIStatusError("boom", response=response, body=None)


class _FakeCompletions:
    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeGroqClient:
    def __init__(self, side_effects):
        self.chat = _FakeChat(_FakeCompletions(side_effects))


def _fake_response(text: str):
    message = type("M", (), {"content": text})()
    choice = type("C", (), {"message": message})()
    return type("R", (), {"choices": [choice]})()


def test_complete_returns_text_on_first_success():
    fake = _FakeGroqClient([_fake_response("hello")])
    client = GroqClient(api_key="x", client=fake, max_retries=3, base_backoff_seconds=0)
    assert client.complete("sys", "user") == "hello"
    assert fake.chat.completions.calls == 1


def test_complete_retries_on_retryable_status_then_succeeds(monkeypatch):
    monkeypatch.setattr("rag.generation.groq_client.time.sleep", lambda s: None)
    fake = _FakeGroqClient([_status_error(429), _fake_response("recovered")])
    client = GroqClient(api_key="x", client=fake, max_retries=3, base_backoff_seconds=0)
    assert client.complete("sys", "user") == "recovered"
    assert fake.chat.completions.calls == 2


def test_complete_raises_immediately_on_non_retryable_status():
    fake = _FakeGroqClient([_status_error(400)])
    client = GroqClient(api_key="x", client=fake, max_retries=3, base_backoff_seconds=0)
    with pytest.raises(APIStatusError):
        client.complete("sys", "user")
    assert fake.chat.completions.calls == 1


def test_complete_raises_runtime_error_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("rag.generation.groq_client.time.sleep", lambda s: None)
    fake = _FakeGroqClient([_status_error(503), _status_error(503), _status_error(503)])
    client = GroqClient(api_key="x", client=fake, max_retries=3, base_backoff_seconds=0)
    with pytest.raises(RuntimeError):
        client.complete("sys", "user")
    assert fake.chat.completions.calls == 3
