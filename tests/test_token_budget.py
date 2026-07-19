from rag.generation.token_budget import count_tokens, fit_to_budget
from rag.models import Chunk, RetrievedChunk


def _retrieved(text: str, idx: int) -> RetrievedChunk:
    chunk = Chunk(
        text=text,
        doc_id="d1",
        chunk_index=idx,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=len(text),
    )
    return RetrievedChunk(chunk=chunk, score=1.0 - idx * 0.01)


def test_count_tokens_positive_for_nonempty_text():
    assert count_tokens("hello world") > 0


def test_fit_keeps_all_chunks_under_budget():
    chunks = [_retrieved("short passage", i) for i in range(3)]
    result = fit_to_budget(chunks, max_tokens=1000)
    assert len(result) == 3


def test_fit_drops_chunks_over_budget():
    long_text = "word " * 500  # ~500 tokens
    chunks = [_retrieved(long_text, i) for i in range(5)]
    result = fit_to_budget(chunks, max_tokens=600)
    assert len(result) < 5
    assert len(result) >= 1


def test_fit_always_keeps_first_chunk_even_if_over_budget():
    huge_text = "word " * 10000
    chunks = [_retrieved(huge_text, 0)]
    result = fit_to_budget(chunks, max_tokens=10)
    assert len(result) == 1


def test_fit_preserves_order():
    chunks = [_retrieved("short passage", i) for i in range(3)]
    result = fit_to_budget(chunks, max_tokens=1000)
    assert [r.chunk.chunk_index for r in result] == [0, 1, 2]


def test_fit_empty_list_returns_empty():
    assert fit_to_budget([], max_tokens=1000) == []
