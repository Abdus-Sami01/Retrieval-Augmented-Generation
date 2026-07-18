import pytest
from pypdf import PdfWriter

from rag.loaders import load_path
from rag.loaders.pdf_loader import PdfLoader
from rag.loaders.text_loader import TextLoader


def test_text_loader_reads_txt(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("some content here", encoding="utf-8")
    doc = TextLoader().load(str(f))
    assert doc.text == "some content here"
    assert doc.doc_type == "text"
    assert doc.source_path == str(f)


def test_text_loader_marks_markdown(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\ncontent", encoding="utf-8")
    doc = TextLoader().load(str(f))
    assert doc.doc_type == "markdown"


def test_text_loader_rejects_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(ValueError):
        TextLoader().load(str(f))


def test_pdf_loader_extracts_text(tmp_path):
    f = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(f, "wb") as fh:
        writer.write(fh)
    # blank page has no extractable text -> loader must raise, not silently return empty doc
    with pytest.raises(ValueError):
        PdfLoader().load(str(f))


def test_pdf_loader_rejects_corrupt_file(tmp_path):
    f = tmp_path / "corrupt.pdf"
    f.write_bytes(b"not a real pdf")
    with pytest.raises(ValueError):
        PdfLoader().load(str(f))


def test_load_path_dispatches_by_suffix(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello", encoding="utf-8")
    doc = load_path(str(f))
    assert doc.text == "hello"


def test_load_path_rejects_unknown_suffix(tmp_path):
    f = tmp_path / "note.docx"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        load_path(str(f))
