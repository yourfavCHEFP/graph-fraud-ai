from fastapi import APIRouter, HTTPException

from deployment.fastapi.schemas import PredictionRequest, PredictionResponse
from deployment.fastapi.services.inference import get_predictor

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return get_predictor().predict_transaction(request.transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Predictor failed to initialize at startup (see app.py's startup
        # event) or was never reached. 503 = "service exists but isn't
        # ready", matching /ready's own status code for the same case --
        # previously this fell through as an unhandled 500.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
