# VitalsAI+ — Brand & Narrative Differentiator Index (BNDI)

AI-driven MarTech platform that scores brand positioning and storytelling
effectiveness across **8 differentiation dimensions** and benchmarks a brand
against its competitors.

This repo is the **Phase 1 prototype**. It ships a runnable core:

- A transparent, formula-based **BNDI scoring engine** (8 metrics + weighted
  overall score + percentile ranking vs. competitors).
- A **FastAPI** service exposing scoring + benchmarking endpoints.
- **Offline sample data** and an offline sentiment analyzer so the whole thing
  runs with **zero API keys and zero network access**.
- A **pytest** suite covering the metric math.

The ML/ensemble layer described in the full technical report
(`../VitalsAI_BNDI_Technical_Implementation_Report.md`) is stubbed behind the
same interface — the heuristic scorer is the drop-in baseline you train against.

## The 8 dimensions

| # | Dimension | What it measures |
|---|-----------|------------------|
| 1 | Emotional Resonance | Depth of emotional connection in messaging |
| 2 | Innovation | Perceived tech advancement & feature differentiation |
| 3 | Narrative Consistency | Coherence of messaging across channels |
| 4 | Market Positioning | Clarity of segment definition & position |
| 5 | Audience Relevance | How well messaging resonates with target audience |
| 6 | Competitor Differentiation | How distinct the brand is vs. competitors |
| 7 | Messaging Effectiveness | How effectively messaging converts/resonates |
| 8 | Trend Alignment | Alignment with current market/cultural trends |

Each score is `0–100`. The **Overall BNDI Score** is a weighted aggregate
(equal weights by default, customizable per industry).

## Quick start (Windows / PowerShell)

```powershell
cd vitals-ai\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
# -> http://127.0.0.1:8000/docs   (interactive Swagger UI)

# Run the tests
pytest -q

# Score the bundled sample brands straight from the CLI (no server)
python -m app.cli
```

## Key endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | Liveness check |
| GET  | `/api/v1/brands` | List sample brands |
| GET  | `/api/v1/brands/{brand_id}` | Get one brand's raw data |
| POST | `/api/v1/brands/{brand_id}/calculate-score` | Full BNDI breakdown |
| GET  | `/api/v1/brands/{brand_id}/benchmarks` | Rank vs. competitors |

## Layout

```
vitals-ai/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app
│   │   ├── cli.py             # offline demo runner
│   │   ├── config.py          # metric weights, settings
│   │   ├── schemas/           # Pydantic models
│   │   ├── ml/                # scoring engine + sentiment
│   │   ├── api/routes/        # brands, scores
│   │   └── data/              # offline sample brands
│   ├── tests/                 # pytest suite
│   └── requirements.txt
├── frontend/                  # (Phase 1 — TBD, see report Part 2.2)
└── README.md
```

## Status

Phase 1, slice 1 of the roadmap (report Part 6): **core scoring engine + API**.
Next slices: real data connectors (Twitter/News/reviews/web), ML model training,
React dashboard.
