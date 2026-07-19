import pytest

from rag.evaluation.external_repos import is_gate_available
from rag.evaluation.sufficiency_crosscheck import cross_check_sufficiency
from tests.test_pipeline import FakeEmbedder, FakeStore, _widget_chunk

pytestmark = pytest.mark.skipif(
    not is_gate_available(), reason="retrieval-verification-gate not checked out as a sibling repo"
)


def test_cross_check_sufficient_when_store_has_relevant_chunk():
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    decision = cross_check_sufficiency(FakeEmbedder(), store, "what are widgets?")
    assert decision["sufficient"] is True
    assert decision["reason"] == "ok"


def test_cross_check_insufficient_when_store_empty():
    store = FakeStore()
    decision = cross_check_sufficiency(FakeEmbedder(), store, "what are widgets?")
    assert decision["sufficient"] is False
    assert decision["reason"] == "no_contexts"
