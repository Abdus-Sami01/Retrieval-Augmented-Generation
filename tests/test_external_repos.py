import pytest

from rag.evaluation.external_repos import is_gate_available, is_harness_available, load_gate, load_harness


def test_load_harness_raises_clear_error_when_repo_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EVAL_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="rag-faithfulness-harness"):
        load_harness()


def test_load_gate_raises_clear_error_when_repo_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EVAL_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="retrieval-verification-gate"):
        load_gate()


def test_is_harness_available_false_when_repo_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EVAL_WORKSPACE_ROOT", str(tmp_path))
    assert is_harness_available() is False


@pytest.mark.skipif(not is_harness_available(), reason="rag-faithfulness-harness not checked out as a sibling repo")
def test_load_harness_exposes_expected_functions():
    harness = load_harness()
    assert hasattr(harness.faithfulness, "extract_claims")
    assert hasattr(harness.faithfulness, "check_faithfulness")
    assert callable(harness.models.nli_fn)


@pytest.mark.skipif(not is_gate_available(), reason="retrieval-verification-gate not checked out as a sibling repo")
def test_load_gate_exposes_expected_functions():
    gate = load_gate()
    assert hasattr(gate.sufficiency, "sufficiency_decision")
