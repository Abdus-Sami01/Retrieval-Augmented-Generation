from rag.evaluation.external_repos import load_harness
from rag.models import Answer


def score_answer_faithfulness(answer: Answer) -> dict:
    """Runs rag-faithfulness-harness's claim-extraction + NLI entailment check against
    an already-generated rag-platform Answer. Reuses that repo's scoring code directly
    rather than reimplementing NLI-based faithfulness checking here."""
    harness = load_harness()
    contexts = [c.chunk.text for c in answer.citations]

    if not contexts:
        return {"claims": [], "supported": [], "faithfulness_rate": None}

    claims = harness.faithfulness.extract_claims(answer.text)
    if not claims:
        return {"claims": [], "supported": [], "faithfulness_rate": None}

    supported = harness.faithfulness.check_faithfulness(claims, contexts, harness.models.nli_fn)
    rate = sum(supported) / len(supported)
    return {"claims": claims, "supported": supported, "faithfulness_rate": rate}
