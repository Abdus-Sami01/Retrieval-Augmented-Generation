import json
import pathlib
import sqlite3

import faiss
import numpy as np

from rag.models import Chunk, RetrievedChunk
from rag.vectorstore.base import VectorStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    row_id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    source_path TEXT NOT NULL,
    section_path TEXT,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL
);
"""


class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int, data_dir: str):
        self.dimension = dimension
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "index.faiss"
        self.db_path = self.data_dir / "chunks.sqlite"

        self._index = faiss.IndexFlatIP(dimension)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        matrix = np.array(vectors, dtype="float32")
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"vector dimension {matrix.shape[1]} does not match store dimension {self.dimension}"
            )

        start_row = self._index.ntotal
        self._index.add(matrix)

        rows = []
        for offset, chunk in enumerate(chunks):
            rows.append(
                (
                    start_row + offset,
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.source_path,
                    chunk.section_path,
                    chunk.char_start,
                    chunk.char_end,
                )
            )
        self._conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (row_id, chunk_id, doc_id, chunk_index, text, source_path, section_path, char_start, char_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if self._index.ntotal == 0:
            return []
        query = np.array([query_vector], dtype="float32")
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results: list[RetrievedChunk] = []
        for score, row_id in zip(scores[0], indices[0]):
            if row_id < 0:
                continue
            cursor = self._conn.execute(
                "SELECT chunk_id, doc_id, chunk_index, text, source_path, section_path, char_start, char_end "
                "FROM chunks WHERE row_id = ?",
                (int(row_id),),
            )
            row = cursor.fetchone()
            if row is None:
                continue
            _, doc_id, chunk_index, text, source_path, section_path, char_start, char_end = row
            chunk = Chunk(
                text=text,
                doc_id=doc_id,
                chunk_index=chunk_index,
                source_path=source_path,
                section_path=section_path,
                char_start=char_start,
                char_end=char_end,
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
        return results

    def count(self) -> int:
        return self._index.ntotal

    def persist(self) -> None:
        faiss.write_index(self._index, str(self.index_path))
        meta = {"dimension": self.dimension, "count": self._index.ntotal}
        (self.data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.exists():
            return
        self._index = faiss.read_index(str(self.index_path))
        self.dimension = self._index.d
