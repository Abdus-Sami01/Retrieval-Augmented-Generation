import pytest

from rag.embedding import SentenceTransformerEmbedder

# small model for fast test runs; production default is BAAI/bge-base-en-v1.5
_TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedder(model_name=_TEST_MODEL)


def test_dimension_is_positive(embedder):
    assert embedder.dimension > 0


def test_embed_documents_returns_one_vector_per_text(embedder):
    vectors = embedder.embed_documents(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == embedder.dimension


def test_embed_documents_empty_list_returns_empty(embedder):
    assert embedder.embed_documents([]) == []


def test_embed_query_returns_single_vector(embedder):
    vector = embedder.embed_query("what is the capital of france?")
    assert len(vector) == embedder.dimension


def test_vectors_are_normalized(embedder):
    import math

    vector = embedder.embed_query("normalization check")
    norm = math.sqrt(sum(x * x for x in vector))
    assert abs(norm - 1.0) < 1e-3


def test_similar_texts_are_closer_than_dissimilar(embedder):
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    v_cat = embedder.embed_documents(["the cat sat on the mat"])[0]
    v_dog = embedder.embed_documents(["a dog rested on the rug"])[0]
    v_stock = embedder.embed_documents(["quarterly stock market earnings report"])[0]

    assert cosine(v_cat, v_dog) > cosine(v_cat, v_stock)
