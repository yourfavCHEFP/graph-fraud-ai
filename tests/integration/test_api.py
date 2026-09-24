from fastapi.testclient import TestClient

from deployment.fastapi import app as fastapi_app
from src.inference.errors import ArtifactMissingError


def test_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready(api_client):
    response = api_client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_predict_requires_transaction_id(api_client):
    response = api_client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_returns_typed_response(api_client):
    response = api_client.post("/predict", json={"transaction_id": 3})

    assert response.status_code == 200
    assert response.json()["transaction_id"] == 3
    assert response.json()["prediction"] == "legitimate"


def test_ready_hides_configuration_error_details(monkeypatch):
    class FailingPredictor:
        @classmethod
        def from_artifacts(cls):
            raise ArtifactMissingError("/private/model/checkpoint.pt")

    monkeypatch.setattr(fastapi_app, "FraudPredictor", FailingPredictor)

    with TestClient(fastapi_app.app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"] == "Model configuration is invalid. See server logs."
    assert "checkpoint.pt" not in response.text
