# Graph Fraud AI Deployment Fix Guide

## Render API

1. Push this repository to GitHub.
2. Create Render Web Service.
3. Settings:
   - Runtime: Docker
   - Root Directory: empty
   - Dockerfile Path: ./Dockerfile
4. Environment variables:

GRAPH_PATH=data/graph/fraud_graph_ready.pt
MODEL_CHECKPOINT_PATH=models/production/graphsage_improved.pt
MODEL_REGISTRY_PATH=models/registry/model_registry.json
ALLOW_ORIGINS=*

5. Deploy.

Test:
curl -X POST https://YOUR-URL.onrender.com/predict \
-H "Content-Type: application/json" \
-d '{"transaction_id":1}'

## Streamlit

Deploy deployment/streamlit/app.py.

Set API URL to (must include the `/predict` path -- the client posts
directly to this URL, so a bare host produces a 405):

https://YOUR-RENDER-URL.onrender.com/predict

Install:
pip install -r deployment/streamlit/requirements.txt

Note: the checkpoint's own tuned `validation_threshold` (currently 0.26)
governs predictions by default -- do not set `FRAUD_THRESHOLD_OVERRIDE`
on the Render side unless you deliberately want production to use a
different threshold than the one the evaluated metrics were computed at.

The currently-recommended primary live demo is Streamlit Community
Cloud (this app, pointed at the Render API above). See
docs/deployment/hosting.md for the full picture across all three
deployment targets (Render/FastAPI, Streamlit Cloud, Hugging Face Space).
