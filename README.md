# rag-platform

Production-grade RAG core (Phase 1): ingest docs -> chunk -> embed -> retrieve -> generate a grounded, cited answer. Refuses to answer when retrieval is insufficient instead of hallucinating.

Every swappable part (embedder, vector store, LLM) sits behind an ABC in `rag/*/base.py`. Swap providers by writing one new adapter class; pipeline code (`rag/pipeline.py`) never changes.

## Stack

- Embeddings: `sentence-transformers` (`BAAI/bge-base-en-v1.5` by default), local, free, CPU-feasible.
- Vector store: FAISS `IndexFlatIP` + SQLite sidecar for chunk text/metadata (`rag/vectorstore/faiss_store.py`).
- Generation: Groq API, `llama-3.3-70b-versatile`, strict "answer only from context, cite [n], or say not found" prompting (`rag/generation/prompt.py`).
- API: FastAPI (`rag/api.py`) - `/health`, `POST /ingest`, `POST /query`.
- UI: Gradio (`rag/ui.py`), thin client over the API.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate       # or source .venv/bin/activate on linux/mac
pip install -r requirements.txt
cp .env.example .env         # fill in GROQ_API_KEY
```

## Run

```bash
uvicorn rag.api:build_app --factory --reload   # API on :8000
python rag/ui.py                                # Gradio UI on :7860, calls the API
```

Ingest via API directly:

```bash
curl -X POST http://127.0.0.1:8000/ingest -F "files=@corpus_sample/RAG_Projects.md"
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What does the faithfulness harness check?"}'
```

## Test

```bash
pytest                                  # full suite, ~50s, downloads a small MiniLM model on first run
RUN_LIVE_GROQ_TESTS=1 pytest tests/test_e2e.py   # also hits the real Groq API (needs a valid key)
```

## Architecture

```
file -> Loader -> Document -> Chunker -> Chunk[] -> Embedder -> vectors -> VectorStore.add()
question -> Embedder.embed_query -> VectorStore.search(k) -> chunks -> prompt -> LLMClient -> Answer{text, citations}
```

Chunking is document-aware: markdown headers become `section_path` boundaries, paragraphs pack into token-budgeted chunks (`tiktoken`-counted) with configurable overlap; oversized single paragraphs get hard-split by token window.

If no retrieved chunk clears `RETRIEVAL_MIN_SCORE`, or the LLM itself signals `NOT_FOUND_IN_CONTEXT`, the pipeline returns an explicit "not enough information" answer with zero citations instead of generating anyway.

## Scope

This is Phase 1 (core RAG) of a larger roadmap. Hybrid/reranking retrieval, production hardening (Docker, rate limiting, caching), a dedicated eval harness, and agentic/multi-hop retrieval are out of scope here and land as later, separate builds in this repo.

Two related repos in this workspace cover eval/safety concerns this repo doesn't: `../rag-faithfulness-harness` (NLI-based faithfulness scoring) and `../retrieval-verification-gate` (sufficiency-gated retrieval). This repo is the general-purpose pipeline; those are narrow research artifacts.
