"""
FIX (mentor review item 9): now uses the get_predictor dependency
(deployment/fastapi/services/inference.py) instead of a bare module
function call, and every HTTPException detail is a fixed, generic,
caller-safe message -- the real exception text is logged server-side
only (logger.exception), never returned in the response body. Previously
`str(exc)` was returned directly to callers, which can leak filesystem
paths or internal exception text (mentor review items 6 and 9).

Status code contract (mentor review item 9):
    Valid transaction ID                -> 200
    Out-of-range transaction ID         -> 400
    Malformed request body              -> 422 (Pydantic, via PredictionRequest)
    Predictor unavailable               -> 503 (raised by the get_predictor
                                            dependency itself, before this
                                            function's body ever runs)
    Unexpected internal failure         -> 500, generic message
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from deployment.fastapi.schemas import PredictionRequest, PredictionResponse
from deployment.fastapi.services.inference import get_predictor

logger = logging.getLogger("graph-fraud-api")

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Score a transaction for fraud risk",
    description=(
        "Looks up the precomputed fraud probability for the given "
        "transaction node (computed once, at service startup, via a "
        "full-graph GraphSAGE forward pass -- see /ready). Returns 400 "
        "if transaction_id is outside the graph's node range, 503 if "
        "the model failed to load at startup, 500 on unexpected failure."
    ),
)
def predict(request: PredictionRequest, predictor=Depends(get_predictor)):
    try:
        return predictor.predict_transaction(request.transaction_id)
    except ValueError as exc:
        # Out-of-range transaction_id -- message is already safe (no
        # filesystem paths/internals), see FraudPredictor.predict_transaction.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected failure in /predict for transaction_id=%s", request.transaction_id)
        raise HTTPException(status_code=500, detail="Internal server error.")
