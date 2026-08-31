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

> **Note:** these metrics were recorded before two corrective fixes:
> (1) a feature-column misalignment bug where the model was trained on
> features whose names did not match their actual values, and (2) the
> switch from a random to a chronological train/val/test split (IEEE-CIS
> is time-ordered; a random split leaks future transactions into
> training). **These numbers are known-stale and pending regeneration**
> after retraining against the corrected pipeline -- see
> `reports/archive/` for how they were produced, and do not treat them
> as representative of the corrected model.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6760 |
| PR-AUC | 0.0821 |
| Precision | 0.1092 |
| Recall | 0.3775 |
| F1 | 0.1694 |

These metrics are the recorded benchmark results in the repository registry; they should not be interpreted as a guarantee of production performance.

## Prediction threshold policy

The deployed model uses the checkpoint's own tuned `validation_threshold`
by default (currently 0.26, not the conventional 0.5). An operator can
override this via the `FRAUD_THRESHOLD_OVERRIDE` environment variable,
but doing so means production predictions will differ from the
threshold the evaluated metrics above were computed at -- leave it unset
unless that's a deliberate, understood tradeoff.

## What "explanation" currently means

The API's `explanation` field is threshold-based reasoning (fraud
probability vs. threshold) plus local graph neighborhood context (the
transaction's directly-connected entity nodes) -- it is **not** formal
feature attribution (e.g. SHAP/integrated gradients). Prediction itself
is a full-graph GraphSAGE forward pass computed ONCE at service startup;
each `/predict` call looks up that node's precomputed probability rather
than running fresh per-request inference.

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

The **primary, currently-recommended live demo is Streamlit Community
Cloud** (`deployment/streamlit/`). A Hugging Face Space
(`deployment/huggingface/`) also runs the same model in-process. The
FastAPI + Render deployment (`deployment/fastapi/`) exists and is
maintained, but treat it as the production-architecture reference
implementation rather than the guaranteed-always-up public link --
Render's free tier is memory-constrained for this model's full-graph
startup cost (see `docs/deployment/production-runbook.md`).

```bash
streamlit run deployment/streamlit/app.py
```

Set `API_URL` when the API is hosted remotely.

## Docker

```bash
docker compose up --build
```

## Deployment prerequisites

The champion checkpoint and graph artifact are committed to this
repository **via Git LFS** (not excluded -- see `.gitattributes`).
A plain `git clone` without Git LFS support installed will check out
small text pointer files instead of the real binaries; verify with:

```bash
head -c 100 models/production/graphsage_improved.pt
head -c 100 data/graph/fraud_graph_ready.pt
```

If either prints `version https://git-lfs.github.com/spec/v1...` instead
of binary content, Git LFS did not actually fetch the real file in that
environment -- the API will fail loudly at startup in that case (see
`src/inference/predictor.py`'s LFS pointer check), not silently.

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
