from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 60
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.35
    data_dir: str = "data"

    use_hybrid_retrieval: bool = False
    use_reranker: bool = False
    use_query_rewriting: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_k: int = 20

    rate_limit_requests_per_minute: int = 60

    query_cache_enabled: bool = False
    query_cache_ttl_seconds: int = 300
    query_cache_max_size: int = 1000
