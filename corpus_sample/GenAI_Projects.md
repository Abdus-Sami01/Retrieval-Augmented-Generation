# Generative AI — Project List

Scope note: "GenAI" as a category is the most saturated of all four — everyone has a fine-tuning notebook and an image-gen wrapper. Your angle should be **measurement and control**: does the generated output actually satisfy a constraint, can you quantify quality/faithfulness, can you make generation more reliable/cheaper/smaller. That's the thread that connects this category back to your verification brand instead of leaving it as a disconnected "I also did some GenAI stuff" section.

## Tier 1 — High leverage, ties to your research brand

1. **Structured-Output Reliability Benchmark + Guardrail Library** — measure how often various open/closed models produce schema-valid JSON under increasingly adversarial prompts, then build a lightweight validate-and-repair loop (schema check → targeted re-prompt → fallback) and show the reliability lift. Directly monetizable — every company with an LLM pipeline has this exact pain point.
2. **Constrained Generation via Z3 Post-Check** — LLM generates a candidate (e.g., a scheduling plan, a math solution, a config file), Z3 formally verifies it satisfies stated constraints, disagreements trigger regeneration with the violated constraint surfaced in the next prompt. Direct extension of your HALO neuro-symbolic verification pipeline into the generation setting rather than the claim-checking setting.
3. **Faithfulness Scorer for Summarization/Generation** — build an NLI-based (entailment) scorer that flags generated content not supported by its source document, benchmark it against human-labeled faithfulness datasets (e.g., XSum-Faith, FRANK). This is measurement-of-generation-quality — squarely your lane.

## Tier 2 — Real-world, deployable problems

4. **Domain-Specific Fine-Tuning with Honest Before/After Eval** — LoRA/QLoRA fine-tune a small open model (Llama-3.2-1B/3B, Qwen-2.5-1.5B class) on a narrow domain (e.g., neon-sign/manufacturing spec generation, or Urdu-English code-switched text), report a proper before/after eval — not just "it feels better," actual metrics. Bread-and-butter consulting skill, most freelancers can't actually demonstrate they did the eval correctly.
5. **Cost/Latency-Optimized Generation Pipeline** — speculative decoding or draft-model + verifier setup, benchmark tokens/sec and cost vs. a single large-model baseline, report the accuracy tradeoff honestly. Directly sellable: "I can cut your generation costs by X% with measured quality impact of Y%."
6. **Prompt Compression / Context Reduction Tool** — given a long prompt/context, compress it (extractive or learned) while measuring downstream task performance retention. Real problem for anyone paying per-token at scale.
7. **Synthetic Data Generator with Quality Filtering** — generate synthetic training data for a narrow task (e.g., low-resource Urdu instruction pairs), then build an automated quality filter (diversity + correctness checks) rather than dumping raw LLM output as a dataset. The filtering step is what separates this from a toy project.

## Tier 3 — Portfolio breadth / fast builds

8. **Style-Controlled Generation with Measurable Style Transfer** — fine-tune or prompt-engineer for a specific style (e.g., your own "zero-abstraction, dense, direct" communication preference — genuinely fun to build since it's your own voice), measure style-match with a classifier rather than eyeballing it.
9. **Hallucination-Rate Comparison Across Open Models** — small, focused benchmark (not a full leaderboard, save that for the agentic/eval category) comparing 3–4 open models' hallucination rates on a fixed closed-book QA set, using your HALO-style claim-extraction approach as the measurement tool. Cheap, and doubles as a HALO demo.
10. **RAG-Free Long-Context vs. RAG Comparison** — for a fixed corpus, compare stuffing everything into a long-context model vs. retrieval-augmented generation on accuracy, cost, and latency — an increasingly relevant real question as context windows grow, and a good "which approach and when" writeup.

## If you build three
1, 2, 4 — a guardrail library (sellable, general-purpose), a Z3-verified generation pipeline (extends HALO directly), and a properly-evaluated fine-tune (proves baseline competency clients actually check for). Covers productized research, research extension, and demonstrable fundamentals.

## Feasibility note
Fine-tuning (#4) needs a free Colab T4/A10 or your Modal setup from RLAT — cheap, not free but low cost (~$5–15 per experiment). Everything else runs on API calls to existing models, no GPU required. Keep synthetic/eval datasets small (hundreds, not tens of thousands of examples) — quality of the eval matters far more than volume for portfolio purposes.
