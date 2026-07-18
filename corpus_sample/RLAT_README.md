# RLAT — Reading Level Adaptive Tutor

Closed loop: child reads aloud → speech model scores fluency → adaptive engine
serves next passage at the Zone-of-Proximal-Development difficulty.

Three components:
- **C1 Speech Fluency Analyzer** — Wav2Vec2 → WCPM / Accuracy / Prosody → Fluency Index.
- **C2 Text Difficulty Scorer** — BERT readability → FAISS-indexed passage bank.
- **C3 Adaptive Engine** — continuous-response IRT (student ability θ) + passage selection.

> Status: **Phase 0 — data**. No model code yet by design. Label availability is
> the gate; we audited it before downloading anything.

## Phase 0 — Label availability audit

```bash
pip install -r requirements.txt
python scripts/run_label_audit.py --json
# -> reports/label_audit.md  (+ .json)
```

The audit grades every dataset in [`data/registry.yaml`](data/registry.yaml)
against the RLAT target schema ([`rlat/data/schema.py`](rlat/data/schema.py)) and
reports, per model head, whether it is trainable **now** (free data),
**if licensed** (LDC in hand), or **never** (dead-end).

### Findings (no LDC licence)

| Head | Now (free) | If LDC | Note |
|---|:--:|:--:|---|
| C2 difficulty | ✅ | ✅ | CommonLit + ReadabilityDB + OneStopEnglish — ship now |
| C1 WCPM | ❌ | ✅ | derivable, but child timed-reading only in MyST/OGI (LDC) |
| C1 Accuracy | ❌ | ✅ | same gate |
| C1 Prosody | ❌ | ❌ | **no rubric labels exist anywhere** |

Three hard facts surfaced:

1. **WCPM/Accuracy are never native** — no corpus ships them. They must be
   *derived* by forced-aligning audio against the reference passage (MFA), then
   counting miscues. So the reference text is mandatory; spontaneous-speech
   corpora (CORAAL, CommonVoice sentences) cannot supply fluency labels.
2. **Child fluency labels are LDC-gated** — only MyST and OGI Kids contain real
   child read-aloud miscues. Free corpora (LibriSpeech) are adult + near-fluent =
   weak disfluency signal. Without LDC, C1 trains on synthetic disfluency + adult
   transfer only.
3. **Prosody is a dead-end** — the NAEP-style expression rubric exists in no
   public corpus. Options: synthetic expressive-vs-flat TTS, weak proxy from
   F0/pause statistics, or pay for human labels. This is the supervision
   bottleneck, not a detail.

### Consequence for build order

- **Start C2 now** — fully unblocked, free data, deliverable this week.
- **C1**: pursue LDC (MyST/OGI) in parallel; meanwhile build the
  forced-alignment → derived-label pipeline and the synthetic-disfluency
  generator so the moment a licence lands, training is one command.
- **Prosody**: descope to a weak proxy head for v1; flag human-label cost in
  the thesis budget.

## Layout

```
data/registry.yaml        dataset metadata (access, native labels, target status)
rlat/data/schema.py       canonical RLAT targets + derivation strategies
rlat/data/audit.py        coverage logic (now / gated / dead-end)
rlat/data/report.py       markdown renderer
scripts/run_label_audit.py
reports/                  generated audit output
```
