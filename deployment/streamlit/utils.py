import os

import requests


def predict_transaction(transaction_id, api_url=None):
    api_url = api_url or os.getenv("API_URL", "http://localhost:8000/predict")
    response = requests.post(
        api_url, json={"transaction_id": int(transaction_id)}, timeout=60
    )
    response.raise_for_status()
    return response.json()
