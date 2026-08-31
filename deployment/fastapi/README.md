# FastAPI Service

Start:

```bash
uvicorn deployment.fastapi.app:app --reload
```

## Endpoints

```text
GET  /health          Liveness only -- process is up, does NOT confirm the model loaded.
GET  /ready           Reflects real model state: 200 if the predictor loaded successfully
                       at startup, 503 if it did not (generic error message only --
                       the real exception is server-log-only, never returned to callers).
POST /predict          Score a transaction. See below. NOT a GET endpoint (a previous
                       version of this file incorrectly listed it as one).
```

## Authentication

None currently implemented. The API is open by default
(`ALLOW_ORIGINS` controls CORS, not authentication). Do not expose this
publicly without adding an auth layer if the deployment is meant to be
access-controlled.

## POST /predict

Request:

```json
{"transaction_id": 12345}
```

Successful response (200):

```json
{
  "transaction_id": 12345,
  "prediction": "fraud",
  "fraud_probability": 0.8734,
  "threshold": 0.26,
  "model": "GraphSAGE",
  "explanation": {
    "risk_level": "high",
    "risk_factors": ["model confidence exceeded fraud threshold"],
    "graph_context": {
      "transaction_id": 12345,
      "neighbor_count": 3,
      "neighbors": [4821, 993, 15602]
    }
  }
}
```

Error responses:

```text
400   transaction_id is out of range for the loaded graph
422   malformed request body (e.g. missing/non-integer transaction_id)
503   model not ready (see /ready for the generic reason)
500   unexpected internal failure (generic message; real exception is
      logged server-side only, never returned to the caller)
```
