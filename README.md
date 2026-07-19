# rag-platform

Production-grade RAG core: ingest docs -> chunk -> embed -> retrieve -> generate a grounded, cited answer. Refuses to answer when retrieval is insufficient instead of hallucinating.

Every swappable part (embedder, vector store, LLM) sits behind an ABC in `rag/*/base.py`. Swap providers by writing one new adapter class; pipeline code (`rag/pipeline.py`) never changes.

## Stack

- Embeddings: `sentence-transformers` (`BAAI/bge-base-en-v1.5` by default), local, free, CPU-feasible.
- Vector store: FAISS `IndexFlatIP` + SQLite sidecar for chunk text/metadata (`rag/vectorstore/faiss_store.py`).
- Generation: Groq API, `llama-3.3-70b-versatile`, strict "answer only from context, cite [n], or say not found" prompting (`rag/generation/prompt.py`).
- API: FastAPI (`rag/api.py`) - `/health`, `POST /ingest`, `POST /query`.
- UI: Gradio (`rag/ui.py`), thin client over the API.
- Retrieval quality (opt-in, all off by default): BM25 keyword index (`rag/retrieval/keyword_index.py`), hybrid dense+BM25 retrieval via reciprocal rank fusion (`rag/retrieval/hybrid.py`), cross-encoder reranking (`rag/retrieval/cross_encoder_reranker.py`), LLM-based query rewriting for vague questions (`rag/retrieval/query_rewriter.py`).

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

### Docker

```bash
docker compose up --build   # API on :8000, Gradio UI on :7860, index persisted in a named volume
```

Reads config from `.env` (same file as local dev - `GROQ_API_KEY` required). The UI container talks to the API container via `RAG_API_BASE_URL=http://api:8000`, set in `docker-compose.yml`.

Note: this Dockerfile/compose setup was authored and statically checked (YAML validity, path/dependency review) but not build-and-run verified - no Docker daemon was available in the environment this was built in. Run `docker compose up --build` and confirm `/health` returns 200 before relying on it.

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

## Production hardening

- Structured JSON logs with per-request latency (`rag/observability.py`) for `/ingest` and `/query`.
- Per-client token-bucket rate limiting (`rag/rate_limit.py`), 429 on exceeded, default 60 req/min.
- Opt-in TTL query cache (`rag/cache.py`) keyed on question+filters.
- Token-budget cap on retrieved context before generation (`rag/generation/token_budget.py`), default 6000 tokens.
- Opt-in PII redaction (email/SSN/phone) at ingest time (`rag/pii.py`).
- Dockerfile + docker-compose for API + UI.

## Evaluation

Rather than duplicate faithfulness/sufficiency scoring, this repo imports the two sibling repos directly (`rag/evaluation/external_repos.py`, sys.path-based - neither is pip-installable) and runs them against rag-platform's own answers:

- `rag/evaluation/faithfulness.py` - reuses `rag-faithfulness-harness`'s claim-extraction + NLI entailment scoring on real `Answer` objects.
- `rag/evaluation/sufficiency_crosscheck.py` - reuses `retrieval-verification-gate`'s independent sufficiency decision as a second opinion, always against a fresh plain-cosine retrieval (gate's threshold is calibrated for that scale, not RRF/reranker scores).
- `rag/evaluation/runner.py` + `eval/golden_queries.json` (14 hand-written queries against `corpus_sample/`, 10 answerable + 4 out-of-scope) compute sufficiency accuracy, retrieval hit-rate, and mean faithfulness rate.
- `python -m rag.evaluation.cli` runs the golden set through **baseline** (plain dense retrieval) and **full** (hybrid + reranker + query-rewriting) configs and fails (exit 1) if full regresses vs baseline on sufficiency accuracy or retrieval hit-rate - the actual "beat the baseline" gate.

**Real run result** (2026-07, `llama-3.3-70b-versatile`): both configs hit `sufficiency_accuracy: 1.0` and `retrieval_hit_rate: 1.0` on the 14-query golden set - full config passes the gate cleanly.

`mean_faithfulness_rate` came back low across the golden-set run (0.18 baseline / 0.20 full) despite the answers looking well-grounded and correctly-cited in every live spot-check this repo was built and tested against. Full root-cause needs fresh per-query data (blocked mid-investigation on Groq's free-tier daily token cap - `RateLimitError: tokens per day` - `groq_client.py`'s retry logic correctly surfaced this rather than hiding it), but the leading hypothesis is confirmed rather than speculative: re-scoring a real, previously-captured answer (the "what does the faithfulness harness check?" query and its actual citations from an earlier live test) directly through `score_answer_faithfulness` gave **0.6**, not ~0.19 - and one of the two "unsupported" claims in that answer was a near-verbatim quote of its cited chunk, marked as NOT entailed anyway. That's `harness.faithfulness.check_faithfulness` requiring a single retrieved chunk to strictly NLI-entail a claim: a general-purpose entailment model scores complex, citation-marker-laden sentences as "neutral" even when the claim is genuinely, almost word-for-word supported. It's a known false-negative mode for NLI-based faithfulness metrics, not evidence of real hallucination - but the 0.6-vs-0.19 gap also shows real per-query variance the aggregate number hides. Reported as measured, not adjusted away. Open item: rerun `python -m rag.evaluation.cli --verbose` once quota resets and inspect all 14 `per_query` entries for the full distribution.

## Scope

This is the core ingest/retrieve/generate pipeline plus retrieval-quality, hardening, and evaluation layers. Agentic/multi-hop retrieval is out of scope here and lands as a later, separate build in this repo.

Two related repos in this workspace cover eval/safety concerns this repo doesn't duplicate: `../rag-faithfulness-harness` (NLI-based faithfulness scoring) and `../retrieval-verification-gate` (sufficiency-gated retrieval) - both are reused directly, see Evaluation above.
