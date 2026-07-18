# Retrieval-Verification Gate

RAG that says "I don't have enough information" instead of confidently generating from irrelevant context.

## Why

Naive RAG always generates, even off garbage retrieval — that's how you get a fluent, wrong answer from a corpus that never had the answer. This gate checks retrieval sufficiency *before* generation and refuses when it fails, on two independent signals: cosine similarity threshold and query/context keyword overlap. Similarity alone is exploitable (embeddings can score two unrelated-but-topically-close sentences highly); keyword overlap alone misses paraphrase. Both together catch more than either.

## How it works

1. `gate/index.py` — FAISS cosine search, returns `(doc_id, score)` pairs.
2. `gate/sufficiency.py` — pure decision function: `no_contexts` / `low_similarity` / `no_keyword_overlap` / `ok`.
3. `gate/answer.py` — extractive answerer (no LLM/API key needed), only invoked when the gate says `ok`.
4. `gate/pipeline.py` — wires retrieval → gate → answer-or-refuse.
5. `gate/models.py` — the only file touching a real model (`all-MiniLM-L6-v2`), lazily loaded. Everything else takes `embed_fn` injected, so tests run in under a second with zero downloads.

## Corpus

Same 20 synthetic handbook docs as the sibling `rag-faithfulness-harness` project. `corpus/eval_queries.json` has 8 in-scope questions (answer exists in the corpus) and 4 out-of-scope questions (weather, sports, geography trivia — nothing the corpus could ever answer) with an `expected_sufficient` label, so the eval directly measures whether the gate refuses when it should.

## Run it

```
pip install -r requirements.txt
python -m pytest tests/ -q                          # 15 unit tests, no model downloads, <1s
python -m gate.cli gate-eval --k 3                   # gate accuracy against eval_queries.json
python -m gate.cli query "What is the capital of France?"   # live refusal example
```

## Result

12/12 gate accuracy on the eval set: every in-scope query answers, every out-of-scope query refuses with `low_similarity` instead of fabricating an answer from the nearest-but-irrelevant chunk.
