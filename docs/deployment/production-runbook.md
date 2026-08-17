# Production Runbook

## Local API

```bash
uvicorn deployment.fastapi.app:app --host 0.0.0.0 --port 8000
```

## Readiness check

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

## Streamlit

Set `API_URL` and run:

```bash
streamlit run deployment/streamlit/app.py
```

## Important
The application requires the champion checkpoint and graph artifact at runtime. They are intentionally excluded from Git by `.gitignore`.
