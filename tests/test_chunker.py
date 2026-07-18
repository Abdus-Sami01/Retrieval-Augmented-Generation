from rag.chunking import DocumentAwareChunker
from rag.models import Document


def test_short_document_produces_single_chunk():
    doc = Document(text="hello world, this is short.", source_path="a.txt", doc_type="text")
    chunker = DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == doc.text.strip()
    assert chunks[0].chunk_id == f"{doc.doc_id}::0"


def test_long_document_splits_into_multiple_chunks():
    paragraph = "This is one paragraph of filler content used to pad token counts. " * 20
    text = "\n\n".join([paragraph] * 10)
    doc = Document(text=text, source_path="b.txt", doc_type="text")
    chunker = DocumentAwareChunker(chunk_size_tokens=100, overlap_tokens=20)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.doc_id == doc.doc_id


def test_consecutive_chunks_actually_overlap():
    paragraph = "Sentence number {} provides unique padding content for testing overlap behavior."
    text = "\n\n".join(paragraph.format(i) for i in range(30))
    doc = Document(text=text, source_path="c.txt", doc_type="text")
    chunker = DocumentAwareChunker(chunk_size_tokens=80, overlap_tokens=20)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    first_tail = chunks[0].text.strip().split("\n\n")[-1]
    second_head = chunks[1].text.strip().split("\n\n")[0]
    assert first_tail == second_head


def test_markdown_headers_produce_section_path():
    text = "# Title\n\nintro text\n\n## Sub\n\nsub content here"
    doc = Document(text=text, source_path="d.md", doc_type="markdown")
    chunker = DocumentAwareChunker(chunk_size_tokens=400, overlap_tokens=60)
    chunks = chunker.chunk(doc)
    paths = [c.section_path for c in chunks]
    assert any(p == "Title" for p in paths)
    assert any(p == "Title > Sub" for p in paths)


def test_oversized_single_paragraph_is_hard_split():
    huge_paragraph = "word " * 2000
    doc = Document(text=huge_paragraph, source_path="e.txt", doc_type="text")
    chunker = DocumentAwareChunker(chunk_size_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        DocumentAwareChunker(chunk_size_tokens=100, overlap_tokens=100)
