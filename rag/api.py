import pathlib
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from rag.chunking import DocumentAwareChunker
from rag.config import Settings
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.generation.groq_client import GroqClient
from rag.pipeline import IngestPipeline, QueryPipeline
from rag.vectorstore.faiss_store import FaissVectorStore


class QueryRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    source_path: str
    section_path: str | None
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sufficient_context: bool
    citations: list[CitationOut]


class IngestFileResult(BaseModel):
    filename: str
    ok: bool
    chunks_added: int
    error: str


class IngestResponse(BaseModel):
    results: list[IngestFileResult]


def build_app(
    settings: Settings | None = None,
    ingest_pipeline: IngestPipeline | None = None,
    query_pipeline: QueryPipeline | None = None,
    store: FaissVectorStore | None = None,
) -> FastAPI:
    """Assemble the FastAPI app. Pass ingest_pipeline/query_pipeline/store to inject
    fakes in tests and skip loading real embedding/LLM models."""
    settings = settings or Settings()

    if ingest_pipeline is None or query_pipeline is None or store is None:
        embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
        store = FaissVectorStore(dimension=embedder.dimension, data_dir=settings.data_dir)
        store.load()
        chunker = DocumentAwareChunker(
            chunk_size_tokens=settings.chunk_size_tokens, overlap_tokens=settings.chunk_overlap_tokens
        )
        llm = GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)

        ingest_pipeline = IngestPipeline(chunker=chunker, embedder=embedder, store=store)
        query_pipeline = QueryPipeline(
            embedder=embedder,
            store=store,
            llm=llm,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
        )

    app = FastAPI(title="rag-platform")

    @app.get("/health")
    def health():
        return {"status": "ok", "indexed_chunks": store.count()}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(files: list[UploadFile]):
        results = []
        for upload in files:
            suffix = pathlib.Path(upload.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                tmp_path = tmp.name
            result = ingest_pipeline.ingest_file(tmp_path)
            results.append(
                IngestFileResult(
                    filename=upload.filename,
                    ok=result.ok,
                    chunks_added=result.chunks_added,
                    error=result.error,
                )
            )
        store.persist()
        return IngestResponse(results=results)

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        answer = query_pipeline.answer(request.question)
        return QueryResponse(
            answer=answer.text,
            sufficient_context=answer.sufficient_context,
            citations=[
                CitationOut(
                    source_path=c.chunk.source_path,
                    section_path=c.chunk.section_path,
                    score=c.score,
                    text=c.chunk.text,
                )
                for c in answer.citations
            ],
        )

    return app
