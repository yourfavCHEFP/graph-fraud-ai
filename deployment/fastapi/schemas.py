from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    transaction_id: int = Field(ge=0)
