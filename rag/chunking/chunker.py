import re

import tiktoken

from rag.models import Chunk, Document

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class _Section:
    __slots__ = ("path", "text", "char_offset")

    def __init__(self, path: str, text: str, char_offset: int):
        self.path = path
        self.text = text
        self.char_offset = char_offset


def _split_into_sections(text: str, doc_type: str) -> list[_Section]:
    if doc_type != "markdown":
        return [_Section(path="", text=text, char_offset=0)]

    lines = text.split("\n")
    sections: list[_Section] = []
    header_stack: list[tuple[int, str]] = []
    buffer_lines: list[str] = []
    section_start_char = 0
    char_cursor = 0

    def flush():
        nonlocal buffer_lines, section_start_char
        if buffer_lines:
            body = "\n".join(buffer_lines)
            if body.strip():
                path = " > ".join(title for _, title in header_stack)
                sections.append(_Section(path=path, text=body, char_offset=section_start_char))
        buffer_lines = []

    for line in lines:
        match = _HEADER_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            header_stack = [h for h in header_stack if h[0] < level]
            header_stack.append((level, title))
            section_start_char = char_cursor
        buffer_lines.append(line)
        char_cursor += len(line) + 1

    flush()
    if not sections:
        sections.append(_Section(path="", text=text, char_offset=0))
    return sections


def _split_paragraphs(text: str) -> list[tuple[str, int]]:
    """Return list of (paragraph_text, char_offset_within_section)."""
    paragraphs = []
    offset = 0
    for para in re.split(r"\n\s*\n", text):
        if para.strip():
            start = text.index(para, offset)
            paragraphs.append((para, start))
            offset = start + len(para)
    return paragraphs


def _hard_split_by_tokens(text: str, max_tokens: int) -> list[str]:
    tokens = _ENCODING.encode(text)
    pieces = []
    for i in range(0, len(tokens), max_tokens):
        pieces.append(_ENCODING.decode(tokens[i : i + max_tokens]))
    return pieces


class DocumentAwareChunker:
    def __init__(self, chunk_size_tokens: int = 400, overlap_tokens: int = 60):
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document: Document) -> list[Chunk]:
        sections = _split_into_sections(document.text, document.doc_type)
        chunks: list[Chunk] = []
        chunk_index = 0

        for section in sections:
            paragraphs = _split_paragraphs(section.text)
            units: list[str] = []
            for para_text, _ in paragraphs:
                if _count_tokens(para_text) > self.chunk_size_tokens:
                    units.extend(_hard_split_by_tokens(para_text, self.chunk_size_tokens))
                else:
                    units.append(para_text)

            if not units:
                continue

            current_parts: list[str] = []
            current_tokens = 0

            def flush_chunk():
                nonlocal current_parts, current_tokens, chunk_index
                if not current_parts:
                    return
                chunk_text = "\n\n".join(current_parts)
                start = document.text.find(current_parts[0])
                end = start + len(chunk_text) if start >= 0 else len(chunk_text)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        doc_id=document.doc_id,
                        chunk_index=chunk_index,
                        source_path=document.source_path,
                        section_path=section.path or None,
                        char_start=max(start, 0),
                        char_end=end,
                    )
                )
                chunk_index += 1

            for unit in units:
                unit_tokens = _count_tokens(unit)
                if current_tokens + unit_tokens > self.chunk_size_tokens and current_parts:
                    flush_chunk()
                    overlap_parts: list[str] = []
                    overlap_tok = 0
                    for prev in reversed(current_parts):
                        prev_tok = _count_tokens(prev)
                        if overlap_tok + prev_tok > self.overlap_tokens and overlap_parts:
                            break
                        overlap_parts.insert(0, prev)
                        overlap_tok += prev_tok
                    current_parts = overlap_parts
                    current_tokens = overlap_tok
                current_parts.append(unit)
                current_tokens += unit_tokens

            flush_chunk()

        return chunks
