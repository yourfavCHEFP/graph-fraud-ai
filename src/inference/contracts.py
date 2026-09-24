"""Validated contracts for inference artifacts and responses."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ProductionModel(BaseModel):
    name: str
    checkpoint: Path


class RegistryDocument(BaseModel):
    production_model: ProductionModel
    deployment_ready: bool | None = None
    deployment_note: str | None = None


class GraphContext(BaseModel):
    transaction_id: int
    neighbor_count: int = Field(ge=0)
    neighbors: list[int]


class Explanation(BaseModel):
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
