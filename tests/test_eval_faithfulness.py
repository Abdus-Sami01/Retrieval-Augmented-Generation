import pytest

from rag.evaluation.external_repos import is_harness_available
from rag.evaluation.faithfulness import score_answer_faithfulness
from rag.models import Answer, Chunk, RetrievedChunk

pytestmark = pytest.mark.skipif(
    not is_harness_available(), reason="rag-faithfulness-harness not checked out as a sibling repo"
)


def _citation(text: str, idx: int = 0) -> RetrievedChunk:
    chunk = Chunk(
        text=text,
        doc_id="d1",
        chunk_index=idx,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=len(text),
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


def test_faithfulness_rate_high_when_answer_matches_context():
    answer = Answer(
        text="Widgets are small mechanical devices used in manufacturing.",
        citations=[_citation("Widgets are small mechanical devices used in manufacturing.")],
        sufficient_context=True,
    )
    result = score_answer_faithfulness(answer)
    assert result["faithfulness_rate"] == 1.0
    assert len(result["claims"]) == 1


def test_faithfulness_rate_lower_when_answer_contradicts_context():
    answer = Answer(
        text="The moon is made entirely of green cheese.",
        citations=[_citation("Widgets are small mechanical devices used in manufacturing.")],
        sufficient_context=True,
    )
    result = score_answer_faithfulness(answer)
    assert result["faithfulness_rate"] < 1.0


def test_faithfulness_score_none_when_no_citations():
    answer = Answer(text="Some answer text.", citations=[], sufficient_context=False)
    result = score_answer_faithfulness(answer)
    assert result["faithfulness_rate"] is None
    assert result["claims"] == []
