from rag.models import Chunk
from rag.retrieval import HybridRetriever, KeywordIndex
from rag.vectorstore import FaissVectorStore


def _chunk(idx: int, text: str, tags: list[str] | None = None) -> Chunk:
    return Chunk(
        text=text,
        doc_id="d1",
        chunk_index=idx,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=len(text),
        tags=tags or [],
    )


def _build(tmp_path):
    dense = FaissVectorStore(dimension=3, data_dir=str(tmp_path / "faiss"))
    keyword = KeywordIndex(data_dir=str(tmp_path / "bm25"))
    return dense, keyword


def test_hybrid_returns_dense_only_hit(tmp_path):
    dense, keyword = _build(tmp_path)
    chunks = [_chunk(0, "widgets are small mechanical devices")]
    dense.add(chunks, [[1.0, 0.0, 0.0]])
    keyword.add(chunks)

    retriever = HybridRetriever(dense, keyword)
    results = retriever.search("unrelated words not in corpus", [1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].chunk.chunk_index == 0


def test_hybrid_returns_keyword_only_hit(tmp_path):
    dense, keyword = _build(tmp_path)
    chunks = [_chunk(0, "widgets are small mechanical devices")]
    dense.add(chunks, [[0.0, 1.0, 0.0]])  # orthogonal to the query vector below
    keyword.add(chunks)

    retriever = HybridRetriever(dense, keyword)
    results = retriever.search("widgets mechanical devices", [1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].chunk.chunk_index == 0


def test_hybrid_ranks_chunk_matched_by_both_signals_highest(tmp_path):
    dense, keyword = _build(tmp_path)
    both_chunk = _chunk(0, "widgets are small mechanical devices used for testing")
    dense_only_chunk = _chunk(1, "completely different filler content here")
    keyword_only_chunk = _chunk(2, "widgets mechanical devices mentioned briefly")
    chunks = [both_chunk, dense_only_chunk, keyword_only_chunk]
    dense.add(
        chunks,
        [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 1.0, 0.0]],
    )
    keyword.add(chunks)

    retriever = HybridRetriever(dense, keyword)
    results = retriever.search("widgets mechanical devices", [1.0, 0.0, 0.0], top_k=3)
    assert results[0].chunk.chunk_index == 0


def test_hybrid_deduplicates_chunk_hit_by_both_retrievers(tmp_path):
    dense, keyword = _build(tmp_path)
    chunks = [_chunk(0, "widgets are small mechanical devices")]
    dense.add(chunks, [[1.0, 0.0, 0.0]])
    keyword.add(chunks)

    retriever = HybridRetriever(dense, keyword)
    results = retriever.search("widgets mechanical devices", [1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1


def test_hybrid_applies_filters_to_both_signals(tmp_path):
    dense, keyword = _build(tmp_path)
    legal_chunk = _chunk(0, "widgets are small mechanical devices", tags=["legal"])
    eng_chunk = _chunk(1, "widgets are small mechanical devices", tags=["engineering"])
    chunks = [legal_chunk, eng_chunk]
    dense.add(chunks, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    keyword.add(chunks)

    retriever = HybridRetriever(dense, keyword)
    results = retriever.search(
        "widgets mechanical devices", [1.0, 0.0, 0.0], top_k=5, filters={"tags": ["legal"]}
    )
    assert len(results) == 1
    assert results[0].chunk.tags == ["legal"]


def test_hybrid_returns_empty_when_no_matches(tmp_path):
    dense, keyword = _build(tmp_path)
    retriever = HybridRetriever(dense, keyword)
    results = retriever.search("anything", [1.0, 0.0, 0.0], top_k=5)
    assert results == []
