from rag.generation.prompt import NO_CONTEXT_MARKER, build_prompt
from rag.models import Chunk, RetrievedChunk


def _retrieved(text: str, source: str = "a.txt") -> RetrievedChunk:
    chunk = Chunk(
        text=text,
        doc_id="d1",
        chunk_index=0,
        source_path=source,
        section_path=None,
        char_start=0,
        char_end=len(text),
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


def test_build_prompt_includes_numbered_passages():
    system, user = build_prompt("what is x?", [_retrieved("x is a widget"), _retrieved("y is unrelated")])
    assert "[1]" in user
    assert "[2]" in user
    assert "x is a widget" in user
    assert "what is x?" in user


def test_build_prompt_system_instructs_no_context_marker():
    system, _ = build_prompt("q", [])
    assert NO_CONTEXT_MARKER in system


def test_build_prompt_handles_empty_retrieval():
    _, user = build_prompt("q", [])
    assert "no passages retrieved" in user
