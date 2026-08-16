from fastapi import APIRouter
from deployment.fastapi.services.inference import get_predictor

router = APIRouter(prefix="/predict")


@router.get("")
def predict():
    return get_predictor().predict()
