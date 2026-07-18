import pytest

from rag.models import Chunk, RetrievedChunk
from rag.retrieval import CrossEncoderReranker


def _retrieved(text: str, idx: int, initial_score: float = 0.5) -> RetrievedChunk:
    chunk = Chunk(
        text=text,
        doc_id="d1",
        chunk_index=idx,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=len(text),
    )
    return RetrievedChunk(chunk=chunk, score=initial_score)


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_rerank_empty_candidates_returns_empty(reranker):
    assert reranker.rerank("what are widgets?", [], top_k=5) == []


def test_rerank_scores_are_in_unit_interval(reranker):
    candidates = [_retrieved("widgets are small mechanical devices", 0)]
    results = reranker.rerank("what are widgets?", candidates, top_k=5)
    assert 0.0 <= results[0].score <= 1.0


def test_rerank_puts_relevant_passage_first(reranker):
    candidates = [
        _retrieved("quarterly financial earnings report for the marketing department", 0),
        _retrieved("widgets are small mechanical devices used in manufacturing", 1),
    ]
    results = reranker.rerank("what are widgets used for?", candidates, top_k=2)
    assert results[0].chunk.chunk_index == 1


def test_rerank_respects_top_k(reranker):
    candidates = [_retrieved(f"passage number {i} about various unrelated topics", i) for i in range(5)]
    results = reranker.rerank("some query", candidates, top_k=2)
    assert len(results) == 2
