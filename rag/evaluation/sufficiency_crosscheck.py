from rag.embedding.base import Embedder
from rag.evaluation.external_repos import load_gate
from rag.vectorstore.base import VectorStore


def cross_check_sufficiency(
    embedder: Embedder,
    store: VectorStore,
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
) -> dict:
    """Independent second opinion via retrieval-verification-gate's sufficiency_decision.

    Always runs a fresh plain-cosine retrieval against the given store rather than
    reusing whatever hybrid/reranker configuration the caller's main QueryPipeline is
    running - gate's similarity_threshold is calibrated for raw cosine similarity
    (measured ~0.7-0.8 for genuine matches), not RRF or cross-encoder sigmoid scores,
    which run on entirely different scales (see rag/pipeline.py's reranker docstring).
    This also keeps the check a true independent verifier rather than trusting the same
    retrieval path being evaluated."""
    gate = load_gate()
    query_vector = embedder.embed_query(query)
    retrieved = store.search(query_vector, top_k=top_k)
    top_score = retrieved[0].score if retrieved else 0.0
    contexts = [r.chunk.text for r in retrieved]
    return gate.sufficiency.sufficiency_decision(top_score, query, contexts, similarity_threshold)
