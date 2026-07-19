import os
import pathlib
import sys

_DEFAULT_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _workspace_root() -> pathlib.Path:
    override = os.environ.get("RAG_EVAL_WORKSPACE_ROOT")
    return pathlib.Path(override) if override else _DEFAULT_WORKSPACE_ROOT


def _ensure_on_path(repo_dir_name: str, package_name: str) -> pathlib.Path:
    repo_path = _workspace_root() / repo_dir_name
    if not (repo_path / package_name).is_dir():
        raise RuntimeError(
            f"expected sibling repo '{repo_dir_name}' with a '{package_name}/' package at "
            f"{repo_path} - this integration reuses that repo's scoring code directly and "
            f"requires it checked out next to rag-platform (or set RAG_EVAL_WORKSPACE_ROOT)"
        )
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_path


def load_harness():
    """Imports the 'harness' package from ../rag-faithfulness-harness. Reused for its
    claim-extraction + NLI entailment faithfulness scoring - not duplicated here."""
    _ensure_on_path("rag-faithfulness-harness", "harness")
    import harness.faithfulness
    import harness.models

    return harness


def load_gate():
    """Imports the 'gate' package from ../retrieval-verification-gate. Reused for its
    independent sufficiency-decision logic as a second opinion alongside rag-platform's
    own min_score/reranker gate."""
    _ensure_on_path("retrieval-verification-gate", "gate")
    import gate.sufficiency

    return gate


def is_harness_available() -> bool:
    try:
        load_harness()
        return True
    except RuntimeError:
        return False


def is_gate_available() -> bool:
    try:
        load_gate()
        return True
    except RuntimeError:
        return False
