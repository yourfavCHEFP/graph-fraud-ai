"""FastAPI entrypoint for Graph Fraud AI."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deployment.fastapi.routes.predict import router
from deployment.fastapi.services.inference import get_predictor

app = FastAPI(
    title="Graph Fraud AI API",
    version="1.0.0",
    description="Transaction-level fraud scoring with a GraphSAGE champion model.",
)


@app.get("/")
def root():
    return {"service": "Graph Fraud AI API", "status": "running", "model": "GraphSAGE"}


allow_origins = [
    origin.strip()
    for origin in os.getenv("ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "graph-fraud-ai"}


@app.get("/ready")
def ready():
    """Return readiness only when model and graph artifacts can be loaded."""
    predictor = get_predictor()
    return {
        "status": "ready",
        "model": predictor.model_name,
        "feature_count": int(predictor.features.shape[1]),
    }
