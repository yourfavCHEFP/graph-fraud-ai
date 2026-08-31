from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    transaction_id: int = Field(ge=0)


class GraphContext(BaseModel):
    transaction_id: int
    neighbor_count: int = Field(ge=0)
    neighbors: list[int]


class Explanation(BaseModel):
    # FIX (mentor review item 10): risk_level and risk_factors' possible
    # values were verified directly against
    # src/explainability/model_explainer.py -- it only ever returns
    # "high"/"low" and one of two fixed strings, so Literal is safe here
    # (not a guess). If model_explainer.py's logic changes to produce
    # other values, this schema will correctly start rejecting them,
    # which is the point: it documents the actual current contract
    # instead of silently accepting anything.
    risk_level: Literal["high", "low"]
    risk_factors: list[str]
    graph_context: GraphContext


class PredictionResponse(BaseModel):
    transaction_id: int
    prediction: Literal["fraud", "legitimate"]
    fraud_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    model: str
    explanation: Explanation
