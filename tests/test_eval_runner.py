from rag.evaluation.golden_set import GoldenQuery
from rag.evaluation.runner import run_eval
from rag.pipeline import QueryPipeline
from tests.test_pipeline import FakeEmbedder, FakeLLM, FakeStore, _widget_chunk


def test_run_eval_reports_perfect_scores_for_matching_pipeline():
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    query_pipeline = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM("Widgets [1]."))
    golden = [GoldenQuery(query="what are widgets?", expected_source_substring="a.txt", expected_sufficient=True)]

    report = run_eval(query_pipeline, golden, score_faithfulness=False)

    assert report.total_queries == 1
    assert report.sufficiency_accuracy == 1.0
    assert report.retrieval_hit_rate == 1.0
    assert report.mean_faithfulness_rate is None
    assert report.per_query[0]["retrieval_hit"] is True


def test_run_eval_detects_sufficiency_mismatch():
    store = FakeStore()  # empty - nothing to retrieve
    query_pipeline = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM("anything"))
    golden = [GoldenQuery(query="what are widgets?", expected_source_substring="a.txt", expected_sufficient=True)]

    report = run_eval(query_pipeline, golden, score_faithfulness=False)

    assert report.sufficiency_accuracy == 0.0
    assert report.per_query[0]["actual_sufficient"] is False
    assert report.per_query[0]["expected_sufficient"] is True


def test_run_eval_detects_wrong_source_as_retrieval_miss():
    store = FakeStore()
    store.chunks.append(_widget_chunk())  # source_path="a.txt"
    query_pipeline = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM("Widgets [1]."))
    golden = [
        GoldenQuery(query="what are widgets?", expected_source_substring="wrong.txt", expected_sufficient=True)
    ]

    report = run_eval(query_pipeline, golden, score_faithfulness=False)

    assert report.retrieval_hit_rate == 0.0
    assert report.per_query[0]["retrieval_hit"] is False


def test_run_eval_handles_correctly_refused_query():
    store = FakeStore()
    query_pipeline = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM("anything"))
    golden = [GoldenQuery(query="out of scope question", expected_source_substring=None, expected_sufficient=False)]

    report = run_eval(query_pipeline, golden, score_faithfulness=False)

    assert report.sufficiency_accuracy == 1.0
    assert report.retrieval_hit_rate == 0.0  # no answerable queries in this set
    assert report.per_query[0]["retrieval_hit"] is None


def test_run_eval_aggregates_across_multiple_queries():
    store = FakeStore()
    store.chunks.append(_widget_chunk())
    query_pipeline = QueryPipeline(embedder=FakeEmbedder(), store=store, llm=FakeLLM("Widgets [1]."))
    golden = [
        GoldenQuery(query="q1", expected_source_substring="a.txt", expected_sufficient=True),
        GoldenQuery(query="q2", expected_source_substring="wrong.txt", expected_sufficient=True),
    ]

    report = run_eval(query_pipeline, golden, score_faithfulness=False)

    assert report.total_queries == 2
    assert report.retrieval_hit_rate == 0.5
