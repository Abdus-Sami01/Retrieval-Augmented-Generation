# RAG Faithfulness Evaluation Harness

Measures whether a RAG pipeline's answers are actually grounded in retrieved context, not just whether retrieval "looks right."

## Why

Most RAG demos stop at "chat with your PDF." This harness treats RAG as a measurement problem: it scores retrieval quality (precision@k, recall@k, MRR) AND generation faithfulness (per-claim NLI entailment against retrieved chunks) side by side, so a hallucinated answer built from a perfectly-retrieved context gets caught.

## How it works

1. `harness/index.py` — dense retrieval over a small corpus (sentence-transformers embeddings + FAISS cosine search).
2. `harness/answer.py` — extractive answerer (no external LLM/API key needed): scores sentences by query-word overlap, returns the top-N.
3. `harness/faithfulness.py` — splits an answer into claims, checks each claim against every retrieved chunk with an NLI model (`entailment` = supported).
4. `harness/eval.py` — orchestrates both evals and reports aggregate + per-item metrics.
5. `harness/models.py` — the only file that touches real models (sentence-transformers `all-MiniLM-L6-v2`, NLI `cross-encoder/nli-deberta-v3-small`), lazily loaded and cached. Every other module takes `embed_fn`/`nli_fn` as injected functions, which is what keeps the test suite fast and model-free.

## Corpus

`corpus/docs.json` — 20 synthetic internal-handbook snippets (self-authored, no scraped/ToS-encumbered data).
`corpus/eval_queries.json` — 10 queries with gold relevant-doc-ids, for retrieval metrics.
`corpus/faithfulness_probes.json` — 10 claim/context pairs, half deliberately fabricated (wrong numbers, inverted facts), to prove the NLI detector actually catches hallucination instead of rubber-stamping everything as faithful.

## Run it

```
pip install -r requirements.txt
python -m pytest tests/ -q                        # 31 unit tests, no model downloads, <1s
python -m harness.cli retrieval-eval --k 3         # precision/recall/MRR against eval_queries.json
python -m harness.cli faithfulness-eval            # detector accuracy against faithfulness_probes.json
python -m harness.cli query "How much annual leave do I accrue per month?"
```

## Design choices

- No API key required — the answerer is extractive, not generative, so the harness runs standalone. Swap `harness/answer.py` for a real LLM call and the faithfulness/retrieval scoring code is unchanged; that's the actual reusable part.
- Every model-touching function is dependency-injected (`embed_fn`, `nli_fn`) so unit tests run in under a second with zero downloads. `harness/models.py` is the single integration seam.
- Faithfulness probes include intentionally false claims specifically to verify the detector rejects them, not just accepts true ones — a detector that says "faithful" to everything is worthless and this is the test that would catch that.
