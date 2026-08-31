"""FastAPI entrypoint for Graph Fraud AI."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from deployment.fastapi.routes.predict import router
from src.inference.predictor import FraudPredictor

logger = logging.getLogger("graph-fraud-api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FIX (mentor review item 7): @app.on_event("startup") is deprecated
    since FastAPI 0.93 -- lifespan is the current pattern. The predictor
    is now stored on app.state (not a module-level global in
    services/inference.py), which makes it injectable/overridable in
    tests (see tests/integration/test_api.py) instead of hidden global
    state that every test run shares.
    """
    try:
        logger.info("Initializing FraudPredictor at startup...")
        app.state.predictor = FraudPredictor()
        app.state.startup_error = None
        logger.info("FraudPredictor ready: %s", app.state.predictor.model_name)
    except Exception:
        # Full traceback logged server-side (logger.exception) -- callers
        # only ever see a generic message via /ready or /predict (see
        # routes/predict.py), never this raw exception text.
        logger.exception("Predictor failed to initialize at startup")
        app.state.predictor = None
        app.state.startup_error = "Model initialization failed. See server logs."
    yield


app = FastAPI(
    title="Graph Fraud AI API",
    version="1.0.0",
    description="Transaction-level fraud scoring with a GraphSAGE champion model.",
    lifespan=lifespan,
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


@app.get("/health", summary="Liveness probe", description="Confirms the process is up. Does NOT confirm the model loaded -- see /ready for that.")
def health():
    """
    Liveness only -- confirms the process is up and responding, NOT that
    the model loaded. Render's actual health-check gating uses /ready
    (see render.yaml's healthCheckPath), which does reflect real model
    state. Kept separate on purpose: a liveness probe that depends on a
    slow/heavy resource (the model) can cause an otherwise-fine process
    to be killed and restarted in a loop by an orchestrator.
    """
    return {"status": "healthy", "service": "graph-fraud-ai"}


@app.get("/ready", summary="Readiness probe", description="Returns 200 if the model/graph loaded successfully at startup, 503 with a generic error otherwise. Used as render.yaml's healthCheckPath.")
def ready(response: Response):
    """
    FIX: previously returned _startup_error verbatim in the response body
    (mentor review item 7 -- "do not return _startup_error verbatim").
    That could leak filesystem paths or internal exception text to any
    caller. The real error is now only ever in the server logs; callers
    get a fixed generic message.
    """
    if app.state.predictor is not None:
        return {"status": "ready", "service": "graph-fraud-ai", "ready": True}
    response.status_code = 503
    return {
        "status": "not_ready",
        "service": "graph-fraud-ai",
        "ready": False,
        "error": app.state.startup_error,
    }
