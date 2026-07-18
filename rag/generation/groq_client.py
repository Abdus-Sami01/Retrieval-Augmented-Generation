import time

from groq import APIStatusError, Groq

from rag.generation.base import LLMClient


class GroqClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_retries: int = 3,
        base_backoff_seconds: float = 1.0,
        client: Groq | None = None,
    ):
        self._client = client if client is not None else Groq(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code not in (429, 500, 502, 503, 504):
                    raise
                if attempt < self._max_retries - 1:
                    time.sleep(self._base_backoff_seconds * (2**attempt))
        raise RuntimeError(f"Groq API failed after {self._max_retries} attempts") from last_error
