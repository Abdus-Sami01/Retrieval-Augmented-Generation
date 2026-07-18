import math

from sentence_transformers import CrossEncoder

from rag.models import RetrievedChunk
from rag.retrieval.reranker_base import Reranker


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        raw_scores = self._model.predict(pairs)
        # Cross-encoder logits are unbounded, not the 0..1 scale the rest of the
        # pipeline's min_score cutoff assumes, so squash through a sigmoid.
        rescored = [
            RetrievedChunk(chunk=c.chunk, score=_sigmoid(float(s)))
            for c, s in zip(candidates, raw_scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
