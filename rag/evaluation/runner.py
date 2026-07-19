from dataclasses import dataclass, field

from rag.evaluation.faithfulness import score_answer_faithfulness
from rag.evaluation.golden_set import GoldenQuery
from rag.pipeline import QueryPipeline


@dataclass
class EvalReport:
    total_queries: int
    sufficiency_accuracy: float
    retrieval_hit_rate: float
    mean_faithfulness_rate: float | None
    per_query: list[dict] = field(default_factory=list)


def run_eval(
    query_pipeline: QueryPipeline, golden_queries: list[GoldenQuery], score_faithfulness: bool = True
) -> EvalReport:
    """Runs every golden query through query_pipeline and reports:
    - sufficiency_accuracy: does answer.sufficient_context match the expected label
      (covers both "answers when it should" and "refuses when it should")
    - retrieval_hit_rate: among expected-answerable queries, was the expected source
      actually cited
    - mean_faithfulness_rate: average NLI-entailment faithfulness rate (via
      rag-faithfulness-harness) across queries that produced citations
    """
    per_query = []
    sufficiency_correct = 0
    hit_count = 0
    answerable_count = 0
    faithfulness_scores: list[float] = []

    for gq in golden_queries:
        answer = query_pipeline.answer(gq.query)
        sufficiency_match = answer.sufficient_context == gq.expected_sufficient
        sufficiency_correct += int(sufficiency_match)

        hit = None
        if gq.expected_sufficient:
            answerable_count += 1
            hit = any(gq.expected_source_substring in c.chunk.source_path for c in answer.citations)
            hit_count += int(hit)

        faithfulness_rate = None
        if score_faithfulness and answer.citations:
            result = score_answer_faithfulness(answer)
            faithfulness_rate = result["faithfulness_rate"]
            if faithfulness_rate is not None:
                faithfulness_scores.append(faithfulness_rate)

        per_query.append(
            {
                "query": gq.query,
                "expected_sufficient": gq.expected_sufficient,
                "actual_sufficient": answer.sufficient_context,
                "sufficiency_match": sufficiency_match,
                "retrieval_hit": hit,
                "faithfulness_rate": faithfulness_rate,
            }
        )

    total = len(golden_queries)
    return EvalReport(
        total_queries=total,
        sufficiency_accuracy=(sufficiency_correct / total) if total else 0.0,
        retrieval_hit_rate=(hit_count / answerable_count) if answerable_count else 0.0,
        mean_faithfulness_rate=(
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
        ),
        per_query=per_query,
    )
