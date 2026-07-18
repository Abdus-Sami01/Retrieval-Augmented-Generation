from rag.generation.base import LLMClient
from rag.retrieval import QueryRewriter


class FakeLLM(LLMClient):
    def __init__(self, response: str | None = None, raise_error: bool = False):
        self._response = response
        self._raise_error = raise_error

    def complete(self, system_prompt, user_prompt):
        if self._raise_error:
            raise RuntimeError("llm unavailable")
        return self._response


def test_rewrite_returns_llm_output():
    rewriter = QueryRewriter(llm=FakeLLM("what is the capital city of France?"))
    result = rewriter.rewrite("capital of france")
    assert result == "what is the capital city of France?"


def test_rewrite_falls_back_to_original_on_empty_response():
    rewriter = QueryRewriter(llm=FakeLLM(""))
    result = rewriter.rewrite("original query")
    assert result == "original query"


def test_rewrite_falls_back_to_original_on_llm_error():
    rewriter = QueryRewriter(llm=FakeLLM(raise_error=True))
    result = rewriter.rewrite("original query")
    assert result == "original query"


def test_rewrite_strips_whitespace():
    rewriter = QueryRewriter(llm=FakeLLM("  cleaned up query  "))
    result = rewriter.rewrite("messy query")
    assert result == "cleaned up query"
