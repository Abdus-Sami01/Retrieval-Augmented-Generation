import json
import pathlib
import re
import threading

from rank_bm25 import BM25Okapi

from rag.filters import passes_filters
from rag.models import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "text": chunk.text,
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "source_path": chunk.source_path,
        "section_path": chunk.section_path,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "tags": chunk.tags,
    }


def _chunk_from_dict(data: dict) -> Chunk:
    return Chunk(
        text=data["text"],
        doc_id=data["doc_id"],
        chunk_index=data["chunk_index"],
        source_path=data["source_path"],
        section_path=data["section_path"],
        char_start=data["char_start"],
        char_end=data["char_end"],
        tags=data["tags"],
    )


class KeywordIndex:
    """BM25 keyword index. rank_bm25 has no incremental-add API, so the index is
    rebuilt lazily from the full in-memory corpus the next time search() runs after
    an add() — cheap at this project's corpus scale, and avoids rebuilding on every
    single add() call during a large batch ingest."""

    def __init__(self, data_dir: str):
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.data_dir / "bm25_corpus.json"

        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._lock:
            self._chunks.extend(chunks)
            self._tokenized.extend(_tokenize(c.text) for c in chunks)
            self._bm25 = None

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[RetrievedChunk]:
        with self._lock:
            if not self._chunks:
                return []
            if self._bm25 is None:
                self._bm25 = BM25Okapi(self._tokenized)

            query_tokens = set(_tokenize(query))
            scores = self._bm25.get_scores(list(query_tokens))
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            results: list[RetrievedChunk] = []
            for idx in ranked_indices:
                # BM25's raw score sign isn't a reliable relevance signal on a tiny corpus
                # (IDF can go non-positive for a term common across few documents even on
                # a genuine match), so use actual token overlap as the real inclusion test.
                if not query_tokens & set(self._tokenized[idx]):
                    continue
                chunk = self._chunks[idx]
                if not passes_filters(chunk.source_path, chunk.tags, filters):
                    continue
                results.append(RetrievedChunk(chunk=chunk, score=float(scores[idx])))
                if len(results) >= top_k:
                    break
            return results

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def persist(self) -> None:
        with self._lock:
            data = [_chunk_to_dict(c) for c in self._chunks]
            self.corpus_path.write_text(json.dumps(data), encoding="utf-8")

    def load(self) -> None:
        with self._lock:
            if not self.corpus_path.exists():
                return
            data = json.loads(self.corpus_path.read_text(encoding="utf-8"))
            self._chunks = [_chunk_from_dict(d) for d in data]
            self._tokenized = [_tokenize(c.text) for c in self._chunks]
            self._bm25 = None
