import argparse
import json
import pathlib
import sys
import tempfile

from rag.config import Settings
from rag.evaluation.golden_set import DEFAULT_GOLDEN_QUERIES_PATH, load_golden_queries
from rag.evaluation.runner import EvalReport, run_eval
from rag.pipeline_factory import build_pipelines

CORPUS_DIR = pathlib.Path(__file__).resolve().parents[2] / "corpus_sample"

BASELINE_OVERRIDES = {"use_hybrid_retrieval": False, "use_reranker": False, "use_query_rewriting": False}
FULL_OVERRIDES = {"use_hybrid_retrieval": True, "use_reranker": True, "use_query_rewriting": True}


def _run_config(overrides: dict, golden_queries: list, data_dir: str, score_faithfulness: bool) -> EvalReport:
    settings = Settings(data_dir=data_dir, **overrides)
    ingest_pipeline, query_pipeline, _ = build_pipelines(settings)
    ingest_pipeline.ingest_files([str(p) for p in CORPUS_DIR.glob("*.md")])
    return run_eval(query_pipeline, golden_queries, score_faithfulness=score_faithfulness)


def _report_summary(name: str, report: EvalReport) -> dict:
    return {
        "config": name,
        "total_queries": report.total_queries,
        "sufficiency_accuracy": round(report.sufficiency_accuracy, 3),
        "retrieval_hit_rate": round(report.retrieval_hit_rate, 3),
        "mean_faithfulness_rate": (
            round(report.mean_faithfulness_rate, 3) if report.mean_faithfulness_rate is not None else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rag-platform's golden eval set against baseline vs full config")
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_QUERIES_PATH))
    parser.add_argument("--skip-faithfulness", action="store_true", help="skip NLI faithfulness scoring (faster)")
    parser.add_argument("--verbose", action="store_true", help="print per-query results")
    args = parser.parse_args()

    golden_queries = load_golden_queries(args.golden_set)
    score_faithfulness = not args.skip_faithfulness

    with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as full_dir:
        baseline_report = _run_config(BASELINE_OVERRIDES, golden_queries, baseline_dir, score_faithfulness)
        full_report = _run_config(FULL_OVERRIDES, golden_queries, full_dir, score_faithfulness)

    summary = {
        "baseline": _report_summary("baseline (plain dense retrieval)", baseline_report),
        "full": _report_summary("full (hybrid + reranker + query-rewriting)", full_report),
    }
    print(json.dumps(summary, indent=2))

    if args.verbose:
        print("\n--- baseline per-query ---")
        print(json.dumps(baseline_report.per_query, indent=2))
        print("\n--- full per-query ---")
        print(json.dumps(full_report.per_query, indent=2))

    regressed = (
        full_report.sufficiency_accuracy < baseline_report.sufficiency_accuracy
        or full_report.retrieval_hit_rate < baseline_report.retrieval_hit_rate
    )
    if regressed:
        print("\nFAIL: full config regressed vs baseline on sufficiency_accuracy or retrieval_hit_rate", file=sys.stderr)
        return 1

    print("\nPASS: full config meets or beats baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
