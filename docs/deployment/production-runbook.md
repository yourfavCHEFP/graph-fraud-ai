# Production Runbook

## Artifact strategy

The champion checkpoint (`models/production/graphsage_improved.pt`) and
graph artifact (`data/graph/fraud_graph_ready.pt`) are committed to this
repository **via Git LFS** -- they are intentionally in Git, not excluded
by `.gitignore`. This section previously stated the opposite; that was
wrong and has been corrected.

**Verify a real binary, not an LFS pointer:**

```bash
head -c 100 models/production/graphsage_improved.pt
head -c 100 data/graph/fraud_graph_ready.pt
```

A real file prints binary garbage. A file that instead prints
`version https://git-lfs.github.com/spec/v1...` is a pointer -- the
deploy environment's `git clone`/`docker build` did not actually pull
LFS content. `src/inference/predictor.py` detects this at startup and
fails with a clear error naming the exact file, rather than a cryptic
pickle error deep inside torch.

## Local API

```bash
uvicorn deployment.fastapi.app:app --host 0.0.0.0 --port 8000
```

## Readiness check

```text
GET /health   -- liveness only: process is up, does NOT confirm the model loaded
GET /ready    -- reflects real model state: 200 if loaded, 503 if startup failed
```

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"transaction_id": 12345}'
```

## Threshold policy

The checkpoint's own `validation_threshold` (currently 0.26) is
authoritative by default. `FRAUD_THRESHOLD_OVERRIDE` is an optional,
intentional override -- leave it unset in normal operation, since
setting it means production no longer matches the threshold the
evaluated metrics were computed at.

## Known limitation: full-graph startup cost

The predictor performs ONE full-graph GraphSAGE forward pass at startup
(computing every transaction's probability once), then serves
`/predict` as an O(1) lookup against that precomputed result -- it does
NOT run fresh per-request subgraph extraction and aggregation. For a
~606K-node graph, that startup pass is memory-intensive enough to
exceed Render's free-tier 512MB limit; this is why the currently
recommended live demo is Streamlit Community Cloud
(`deployment/streamlit/`) or the Hugging Face Space
(`deployment/huggingface/`), not the Render-hosted API. See
`src/inference/predictor.py`'s memory-cleanup comments for the specific
mitigations already in place (freeing the model/graph/feature tensors
immediately after the one-time forward pass).

## Streamlit

```bash
streamlit run deployment/streamlit/app.py
```

Set `API_URL` (must include the `/predict` path, e.g.
`https://your-service.onrender.com/predict`) when pointing at a remote
API rather than running one locally.
