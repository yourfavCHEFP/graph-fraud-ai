import pytest
from fastapi.testclient import TestClient

from deployment.fastapi import app as fastapi_app


class FakePredictor:
    model_name = "GraphSAGE"

    @classmethod
    def from_artifacts(cls):
        return cls()

    def predict_transaction(self, transaction_id: int):
        return {
            "transaction_id": transaction_id,
            "prediction": "legitimate",
            "fraud_probability": 0.2,
            "threshold": 0.5,
            "model": self.model_name,
            "explanation": {
                "risk_level": "low",
                "risk_factors": ["model confidence below fraud threshold"],
                "graph_context": {
                    "transaction_id": transaction_id,
                    "neighbor_count": 0,
                    "neighbors": [],
                },
            },
        }


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setattr(fastapi_app, "FraudPredictor", FakePredictor)

    with TestClient(fastapi_app.app) as client:
        yield client
