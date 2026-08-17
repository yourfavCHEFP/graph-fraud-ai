from fastapi.testclient import TestClient

from deployment.fastapi.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_requires_transaction_id():
    response = client.post("/predict", json={})
    assert response.status_code == 422
