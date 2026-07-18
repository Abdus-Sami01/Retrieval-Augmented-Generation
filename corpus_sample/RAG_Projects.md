# RAG (Retrieval-Augmented Generation) — Project List

Scope note: you already have two RAG repos (FAISS CLI chatbot, Nexus-Insight). Do not build a fifth vanilla "chat with your PDF" project — it will read as regression, not growth. Every item below either (a) adds a rigorous evaluation layer vanilla RAG repos skip, or (b) solves a specific real-world retrieval failure mode. The organizing idea: **RAG is a measurement problem as much as a retrieval problem** — most RAG repos never check whether the answer is actually grounded in what was retrieved.

## Tier 1 — High leverage, ties to your research brand

1. **RAG Faithfulness Evaluation Harness** — for any RAG pipeline, automatically score whether the generated answer is entailed by the retrieved chunks (NLI-based), flag unsupported claims, and report a faithfulness rate across a benchmark query set. This is the single most valuable RAG artifact you could build — it's the thing that turns your two existing RAG repos into "I built RAG systems AND I can prove whether they hallucinate," which almost nobody else can say.
2. **Retrieval-Verification Gate** — a RAG variant where retrieved chunks are checked for actual relevance/sufficiency (via a lightweight classifier or entailment check) *before* generation; if retrieval is insufficient, the system explicitly says "I don't have enough information" instead of generating anyway. Directly demonstrates your verification-gap thesis applied to the most common LLM deployment pattern in industry.
3. **Consolidated RAG Repo with Full Eval Suite** — take your existing FAISS chatbot + Nexus-Insight, merge into one polished repo, add retrieval metrics (precision@k, recall@k, MRR) AND generation metrics (faithfulness, answer relevance) side by side. Archive the older duplicates. This alone meaningfully upgrades your existing profile without new build time on the retrieval side.

## Tier 2 — Real-world, deployable problems

4. **Multi-Hop RAG for Regulatory/Compliance Documents** — RAG over documents requiring chained retrieval (answer to question A informs the query for question B), e.g., tax code, technical standards, or legal text. Multi-hop is a known hard failure mode for naive RAG; solving it well is a real consulting-grade skill.
5. **RAG over Structured + Unstructured Mixed Sources** — retrieval that spans a SQL database and a document corpus simultaneously (agent decides which source to query, or a hybrid retriever merges both), useful for the exact kind of business data companies actually have (half spreadsheets, half PDFs).
6. **Freshness-Aware RAG** — retrieval system that handles conflicting/superseded information correctly (e.g., a policy changed last month, older documents shouldn't win), with explicit recency-weighting and conflict detection between retrieved chunks. Real, underserved problem — most RAG demos assume a static, non-contradictory corpus.
7. **Low-Resource Language RAG (Urdu)** — RAG pipeline with an Urdu-capable embedding model, tested on Urdu queries against Urdu/English mixed documents. Genuinely underserved niche, directly usable for your own consulting market.

## Tier 3 — Portfolio breadth / fast builds

8. **Chunking Strategy Ablation Study** — fixed corpus, fixed generator, vary only the chunking strategy (fixed-size, semantic, sentence-window, hierarchical) and embedding model, report retrieval quality deltas. Cheap to build, and a properly-run ablation study is rarer than it should be in public RAG repos — most people pick one chunking method and never test alternatives.
9. **Reranker Impact Study** — same corpus, compare naive top-k retrieval vs. retrieve-then-rerank (cross-encoder reranker), quantify the precision improvement and the added latency cost — an honest tradeoff writeup, not just "reranking is better."
10. **RAG Cost/Latency Optimizer** — for a fixed accuracy target, sweep chunk size, k, embedding model size, and generator size to find the cheapest pipeline configuration that still hits the target — directly answers the question every client actually has ("how do I make this RAG system cheaper without breaking it").

## If you build three
1, 2, 3 — a faithfulness harness (your strongest possible RAG artifact), a verification-gated retrieval pipeline (extends your research thesis), and a consolidation of your existing repos with proper eval bolted on (fixes a real weakness cheaply). This set turns "has built RAG systems" into "has built RAG systems and can prove when they're lying," which is the actual differentiator in this crowded space.

## Feasibility note
All buildable with a small self-constructed corpus (50–200 documents is plenty for a demo — don't over-invest in corpus size) and free-tier embedding models (e.g., open sentence-transformers) plus API calls for generation. No GPU strictly required; a vector DB like FAISS or Chroma runs fine locally. The faithfulness/NLI scoring (#1) can reuse a small open NLI model run on CPU — this is the cheapest project on the whole list relative to its portfolio value.
