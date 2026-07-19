import json

from rag.evaluation.golden_set import DEFAULT_GOLDEN_QUERIES_PATH, GoldenQuery, load_golden_queries


def test_default_golden_queries_file_loads():
    queries = load_golden_queries()
    assert len(queries) >= 10
    assert all(isinstance(q, GoldenQuery) for q in queries)


def test_default_golden_queries_has_both_answerable_and_refusal_cases():
    queries = load_golden_queries()
    assert any(q.expected_sufficient for q in queries)
    assert any(not q.expected_sufficient for q in queries)


def test_refusal_cases_have_no_expected_source():
    queries = load_golden_queries()
    for q in queries:
        if not q.expected_sufficient:
            assert q.expected_source_substring is None


def test_answerable_cases_have_expected_source():
    queries = load_golden_queries()
    for q in queries:
        if q.expected_sufficient:
            assert q.expected_source_substring is not None


def test_load_golden_queries_from_custom_path(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps([{"query": "q1", "expected_source_substring": "a.md", "expected_sufficient": True}]),
        encoding="utf-8",
    )
    queries = load_golden_queries(custom)
    assert len(queries) == 1
    assert queries[0].query == "q1"


def test_default_path_points_at_eval_directory():
    assert DEFAULT_GOLDEN_QUERIES_PATH.name == "golden_queries.json"
    assert DEFAULT_GOLDEN_QUERIES_PATH.exists()
