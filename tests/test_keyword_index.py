from rag.models import Chunk
from rag.retrieval import KeywordIndex


def _chunk(idx: int, text: str, source_path: str = "a.txt", tags: list[str] | None = None) -> Chunk:
    return Chunk(
        text=text,
        doc_id="d1",
        chunk_index=idx,
        source_path=source_path,
        section_path=None,
        char_start=0,
        char_end=len(text),
        tags=tags or [],
    )


def test_search_on_empty_index_returns_empty_list(tmp_path):
    index = KeywordIndex(data_dir=str(tmp_path))
    assert index.search("widgets", top_k=5) == []


def test_add_and_count(tmp_path):
    index = KeywordIndex(data_dir=str(tmp_path))
    index.add([_chunk(0, "widgets are small mechanical devices")])
    assert index.count() == 1


def test_search_ranks_exact_keyword_match_first(tmp_path):
    index = KeywordIndex(data_dir=str(tmp_path))
    index.add(
        [
            _chunk(0, "widgets are small mechanical devices used for testing purposes"),
            _chunk(1, "quarterly financial earnings report for the marketing department"),
        ]
    )
    results = index.search("widgets mechanical devices", top_k=2)
    assert results
    assert results[0].chunk.chunk_index == 0


def test_search_excludes_non_matching_terms(tmp_path):
    index = KeywordIndex(data_dir=str(tmp_path))
    index.add(
        [
            _chunk(0, "widgets are small mechanical devices"),
            _chunk(1, "quarterly financial earnings report"),
        ]
    )
    results = index.search("financial earnings quarterly", top_k=5)
    assert all(r.chunk.chunk_index == 1 for r in results)


def test_search_applies_source_path_filter(tmp_path):
    index = KeywordIndex(data_dir=str(tmp_path))
    index.add(
        [
            _chunk(0, "widgets are small mechanical devices", source_path="docs/policy.md"),
            _chunk(1, "widgets are small mechanical devices", source_path="docs/other.md"),
        ]
    )
    results = index.search("widgets", top_k=5, filters={"source_path": "policy"})
    assert len(results) == 1
    assert results[0].chunk.source_path == "docs/policy.md"


def test_search_applies_tags_filter(tmp_path):
    index = KeywordIndex(data_dir=str(tmp_path))
    index.add(
        [
            _chunk(0, "widgets are small mechanical devices", tags=["legal"]),
            _chunk(1, "widgets are small mechanical devices", tags=["engineering"]),
        ]
    )
    results = index.search("widgets", top_k=5, filters={"tags": ["legal"]})
    assert len(results) == 1
    assert results[0].chunk.tags == ["legal"]


def test_persist_and_reload_preserves_search(tmp_path):
    index = KeywordIndex(data_dir=str(tmp_path))
    index.add([_chunk(0, "widgets are small mechanical devices", tags=["legal"])])
    index.persist()

    reloaded = KeywordIndex(data_dir=str(tmp_path))
    reloaded.load()
    assert reloaded.count() == 1
    results = reloaded.search("widgets", top_k=1)
    assert results[0].chunk.tags == ["legal"]


def test_add_after_search_rebuilds_index(tmp_path):
    index = KeywordIndex(data_dir=str(tmp_path))
    index.add([_chunk(0, "widgets are small mechanical devices")])
    index.search("widgets", top_k=5)  # builds the bm25 index
    index.add([_chunk(1, "widgets widgets widgets everywhere in this document")])
    results = index.search("widgets", top_k=5)
    assert len(results) == 2
