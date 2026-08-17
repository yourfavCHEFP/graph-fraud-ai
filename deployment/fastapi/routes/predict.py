from fastapi import APIRouter
from deployment.fastapi.schemas import PredictionRequest, PredictionResponse
from deployment.fastapi.services.inference import get_predictor

router = APIRouter(prefix="/predict")


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return get_predictor().predict_transaction(
        request.transaction_id
    )
