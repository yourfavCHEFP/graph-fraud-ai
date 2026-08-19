"""FastAPI entrypoint for Graph Fraud AI."""

import logging
import os

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from deployment.fastapi.routes.predict import router
from deployment.fastapi.services.inference import initialize_predictor

logger = logging.getLogger("graph-fraud-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Graph Fraud AI API",
    version="1.0.0",
    description="Transaction-level fraud scoring with a GraphSAGE champion model.",
)

_predictor_ready = False
_startup_error = None


@app.on_event("startup")
def startup_event():
    """
    FIX: the model/graph used to load lazily on the FIRST /predict request.
    Loading here instead means: (1) it only happens once, not per-request,
    and (2) if it fails or is too slow for Render's box, that shows up
    clearly in the DEPLOY logs (which Render already surfaces prominently)
    instead of as a silent 502 on every live prediction request with
    nothing useful in the logs.
    """
    global _predictor_ready, _startup_error
    try:
        initialize_predictor()
        _predictor_ready = True
    except Exception as exc:
        # Logged with full traceback (logger.exception, not logger.error)
        # so Render's log stream shows the REAL cause -- this directly
        # addresses "Render logs hide the real error."
        logger.exception("Predictor failed to initialize at startup")
        _startup_error = str(exc)


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
def ready(response: Response):
    """
    FIX: this used to always return HTTP 200 (with ready:true hardcoded)
    regardless of whether the model/graph actually loaded. Render's
    healthCheckPath gating (see render.yaml) only works if a broken
    instance returns a non-2xx status -- a 200 with {"ready": false} in
    the body is invisible to that mechanism, since Render only looks at
    the status code, not the JSON content.
    """
    if _predictor_ready:
        return {"status": "ready", "service": "graph-fraud-ai", "ready": True}
    response.status_code = 503
    return {
        "status": "not_ready",
        "service": "graph-fraud-ai",
        "ready": False,
        "error": _startup_error,
    }
