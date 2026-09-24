import torch

from src.inference.contracts import PredictionResponse
from src.inference.predictor import FraudPredictor


def test_predictor_import():
    assert FraudPredictor is not None


def test_predictor_can_be_constructed_without_artifacts():
    predictor = FraudPredictor(
        probabilities=torch.tensor([0.1, 0.9]),
        adjacency_index={0: [1], 1: [0]},
        threshold=0.5,
        num_nodes=2,
        model_name="GraphSAGE",
    )

    result = predictor.predict_transaction(1)

    assert isinstance(result, PredictionResponse)
    assert result.prediction == "fraud"
