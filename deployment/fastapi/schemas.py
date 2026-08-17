from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    transaction_id: int = Field(ge=0)


class PredictionResponse(BaseModel):
    transaction_id: int
    prediction: str
    fraud_probability: float
    threshold: float
    model: str
    explanation: dict[str, Any]
