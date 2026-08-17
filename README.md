# Graph Fraud AI

A production-oriented graph neural network fraud detection system built on the IEEE-CIS Fraud Detection dataset. The system models transaction relationships, engineers leakage-safe graph features, benchmarks GCN/GAT/GraphSAGE, selects GraphSAGE as the champion, and exposes transaction-level inference through FastAPI and Streamlit.

## Architecture

```text
IEEE-CIS data
    ↓
Preprocessing
    ↓
Graph construction
    ↓
16 leakage-safe graph features (Phase 10.9)
    ↓
GCN / GAT / GraphSAGE benchmark
    ↓
GraphSAGE champion
    ↓
Inference + explainability
    ↓
FastAPI
    ↓
Streamlit / hosted demo
```

## Champion

- Architecture: GraphSAGE
- Registry: `models/registry/model_registry.json`
- Checkpoint: `models/graphsage_improved.pt`
- Feature pipeline: Phase 10.9
- Feature count: 16

## Test performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6760 |
| PR-AUC | 0.0821 |
| Precision | 0.1092 |
| Recall | 0.3775 |
| F1 | 0.1694 |

These metrics are the recorded benchmark results in the repository registry; they should not be interpreted as a guarantee of production performance.

## API

```bash
uvicorn deployment.fastapi.app:app --host 0.0.0.0 --port 8000
```

Health:

```text
GET /health
GET /ready
POST /predict
```

Prediction request:

```json
{"transaction_id": 12345}
```

## Dashboard

```bash
streamlit run deployment/streamlit/app.py
```

Set `API_URL` when the API is hosted remotely.

## Docker

```bash
docker compose up --build
```

## Deployment prerequisites

The source repository intentionally excludes large runtime artifacts. Before starting the API, provide:

- `models/graphsage_improved.pt`
- `data/graph/fraud_graph_ready.pt`

Run the source-level audit with:

```bash
python scripts/check_deployment.py
```

## Project phases

- Phase 10.9 — leakage-safe graph feature engineering
- Phase 11 — GNN benchmarking and GraphSAGE champion selection
- Phase 12 — production inference layer
- Phase 13 — FastAPI serving
- Phase 14 — deployment foundation
- Phase 15 — explainability and investigation
- Phase 16 — inference/API integration
- Phase 17–20 — production hardening, reliability, deployment configuration, and portfolio readiness

## Repository principles

- No fraud labels are used to construct graph features.
- No global dataset statistics are used in the feature pipeline.
- Model selection is based on validation evidence, with PR-AUC emphasized for the imbalanced fraud task.
- Runtime secrets belong in environment variables, never in Git.
