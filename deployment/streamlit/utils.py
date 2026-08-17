import requests


def predict_transaction(transaction_id, api_url="http://localhost:8000/predict"):
    response = requests.post(
        api_url,
        json={"transaction_id": transaction_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
