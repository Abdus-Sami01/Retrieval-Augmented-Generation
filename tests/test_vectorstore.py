from concurrent.futures import ThreadPoolExecutor

from rag.models import Chunk
from rag.vectorstore import FaissVectorStore


def _chunk(idx: int, doc_id: str = "d1") -> Chunk:
    return Chunk(
        text=f"chunk text {idx}",
        doc_id=doc_id,
        chunk_index=idx,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=10,
    )


def test_add_and_count(tmp_path):
    store = FaissVectorStore(dimension=3, data_dir=str(tmp_path))
    store.add([_chunk(0)], [[1.0, 0.0, 0.0]])
    assert store.count() == 1


def test_search_returns_most_similar_first(tmp_path):
    store = FaissVectorStore(dimension=3, data_dir=str(tmp_path))
    store.add(
        [_chunk(0), _chunk(1), _chunk(2)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
    )
    results = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_index == 0
    assert results[0].score >= results[1].score


def test_search_on_empty_store_returns_empty_list(tmp_path):
    store = FaissVectorStore(dimension=3, data_dir=str(tmp_path))
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


def test_add_rejects_mismatched_lengths(tmp_path):
    import pytest

    store = FaissVectorStore(dimension=3, data_dir=str(tmp_path))
    with pytest.raises(ValueError):
        store.add([_chunk(0), _chunk(1)], [[1.0, 0.0, 0.0]])


def test_add_rejects_wrong_dimension(tmp_path):
    import pytest

    store = FaissVectorStore(dimension=3, data_dir=str(tmp_path))
    with pytest.raises(ValueError):
        store.add([_chunk(0)], [[1.0, 0.0]])


def test_persist_and_reload_preserves_search(tmp_path):
    store = FaissVectorStore(dimension=3, data_dir=str(tmp_path))
    store.add([_chunk(0), _chunk(1)], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    store.persist()

    reloaded = FaissVectorStore(dimension=3, data_dir=str(tmp_path))
    reloaded.load()
    assert reloaded.count() == 2
    results = reloaded.search([1.0, 0.0, 0.0], top_k=1)
    assert results[0].chunk.chunk_index == 0


def test_search_works_from_a_different_thread_than_construction(tmp_path):
    # Regression: FastAPI runs sync endpoints in a threadpool, so the store is built
    # on the app's main thread but queried from worker threads. A sqlite3 connection
    # opened with default check_same_thread=True raises ProgrammingError in that case.
    store = FaissVectorStore(dimension=3, data_dir=str(tmp_path))
    store.add([_chunk(0)], [[1.0, 0.0, 0.0]])

    with ThreadPoolExecutor(max_workers=1) as pool:
        results = pool.submit(store.search, [1.0, 0.0, 0.0], 1).result()

    assert len(results) == 1
    assert results[0].chunk.chunk_index == 0
