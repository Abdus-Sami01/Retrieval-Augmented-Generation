from rag.models import RetrievedChunk
from rag.retrieval.keyword_index import KeywordIndex
from rag.vectorstore.base import VectorStore


class HybridRetriever:
    """Combines dense (FAISS cosine) and keyword (BM25) retrieval via Reciprocal Rank
    Fusion. RRF merges by rank rather than raw score, which sidesteps the fact that
    cosine similarity (0..1) and BM25 scores (unbounded, corpus-dependent) live on
    incomparable scales."""

    def __init__(
        self,
        dense_store: VectorStore,
        keyword_index: KeywordIndex,
        dense_top_k: int = 20,
        keyword_top_k: int = 20,
        rrf_k: int = 60,
    ):
        self.dense_store = dense_store
        self.keyword_index = keyword_index
        self.dense_top_k = dense_top_k
        self.keyword_top_k = keyword_top_k
        self.rrf_k = rrf_k

    def search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        dense_results = self.dense_store.search(query_vector, self.dense_top_k, filters=filters)
        keyword_results = self.keyword_index.search(query_text, self.keyword_top_k, filters=filters)

        rrf_scores: dict[str, float] = {}
        chunk_by_id = {}

        for rank, retrieved in enumerate(dense_results, start=1):
            chunk_id = retrieved.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_by_id[chunk_id] = retrieved.chunk

        for rank, retrieved in enumerate(keyword_results, start=1):
            chunk_id = retrieved.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_by_id[chunk_id] = retrieved.chunk

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [
            RetrievedChunk(chunk=chunk_by_id[cid], score=rrf_scores[cid]) for cid in ranked_ids[:top_k]
        ]
