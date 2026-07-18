from rag.models import Answer, Chunk, Document, RetrievedChunk


def test_document_holds_text_and_metadata():
    doc = Document(text="hello world", source_path="a.txt", doc_type="text")
    assert doc.text == "hello world"
    assert doc.source_path == "a.txt"
    assert doc.doc_type == "text"
    assert doc.doc_id  # auto-generated, non-empty


def test_document_id_is_stable_for_same_source():
    d1 = Document(text="x", source_path="a.txt", doc_type="text")
    d2 = Document(text="x", source_path="a.txt", doc_type="text")
    assert d1.doc_id == d2.doc_id


def test_chunk_holds_text_and_provenance():
    doc = Document(text="hello world", source_path="a.txt", doc_type="text")
    chunk = Chunk(
        text="hello",
        doc_id=doc.doc_id,
        chunk_index=0,
        source_path="a.txt",
        section_path="intro",
        char_start=0,
        char_end=5,
    )
    assert chunk.text == "hello"
    assert chunk.chunk_id == f"{doc.doc_id}::0"
    assert chunk.tags == []


def test_chunk_accepts_explicit_tags():
    chunk = Chunk(
        text="hello",
        doc_id="d1",
        chunk_index=0,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=5,
        tags=["policy", "2026"],
    )
    assert chunk.tags == ["policy", "2026"]


def test_retrieved_chunk_wraps_chunk_with_score():
    chunk = Chunk(
        text="hello",
        doc_id="d1",
        chunk_index=0,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=5,
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.87)
    assert retrieved.chunk.text == "hello"
    assert retrieved.score == 0.87


def test_answer_holds_text_and_citations():
    chunk = Chunk(
        text="hello",
        doc_id="d1",
        chunk_index=0,
        source_path="a.txt",
        section_path=None,
        char_start=0,
        char_end=5,
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.9)
    answer = Answer(text="The answer is X.", citations=[retrieved], sufficient_context=True)
    assert answer.text == "The answer is X."
    assert answer.citations[0].chunk.chunk_id == "d1::0"
    assert answer.sufficient_context is True
