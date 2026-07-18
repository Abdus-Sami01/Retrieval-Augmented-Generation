from sentence_transformers import SentenceTransformer

from rag.embedding.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE models are trained with this instruction prefix on queries only.
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        vector = self._model.encode(
            [prefixed], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return vector.tolist()
