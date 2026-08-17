import json
import torch

from src.models.gnn_model import FraudGraphSAGE
from src.features.graph_features import build_graph_features
from src.explainability.fraud_reasoner import build_fraud_explanation


class FraudPredictor:

    def __init__(
        self,
        registry_path="models/registry/model_registry.json",
        graph_path="data/graph/fraud_graph_ready.pt",
    ):
        self.graph_path = graph_path

        with open(registry_path) as file:
            self.registry = json.load(file)

        checkpoint = torch.load(
            self.registry["production_model"]["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )

        self.threshold = checkpoint["validation_threshold"]

        self.model = FraudGraphSAGE(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
            output_dim=2,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.mean = checkpoint["normalization_mean"]
        self.std = checkpoint["normalization_std"]


    def predict_transaction(self, transaction_id):

        graph = torch.load(
            self.graph_path,
            map_location="cpu",
            weights_only=False,
        )

        features = build_graph_features(graph)
        features = (features - self.mean) / self.std

        with torch.no_grad():
            logits = self.model(
                features,
                graph.edge_index,
            )

            probability = torch.softmax(
                logits,
                dim=1,
            )[transaction_id, 1].item()

        return {
            "transaction_id": transaction_id,
            "prediction": "fraud" if probability >= self.threshold else "legitimate",
            "fraud_probability": round(probability, 6),
            "threshold": self.threshold,
            "model": "GraphSAGE",
            "explanation": build_fraud_explanation(
                graph,
                transaction_id,
                probability,
                self.threshold,
            ),
        }
