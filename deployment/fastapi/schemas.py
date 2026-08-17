from pydantic import BaseModel


class PredictionRequest(BaseModel):
    transaction_id: int


class PredictionResponse(BaseModel):
    transaction_id: int
    prediction: str
    fraud_probability: float
    threshold: float
    model: str
    explanation: dict
