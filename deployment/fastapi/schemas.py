from pydantic import BaseModel


class PredictionResponse(BaseModel):
    count: int
    threshold: float
    fraud_predictions: int
