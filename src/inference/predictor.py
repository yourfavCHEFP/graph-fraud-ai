import json
import torch

from src.models.gnn_model import FraudGraphSAGE
from src.features.graph_features import build_graph_features


class FraudPredictor:

    def __init__(
        self,
        registry_path="models/registry/model_registry.json",
        graph_path="data/graph/fraud_graph_ready.pt",
    ):
        self.registry_path = registry_path
        self.graph_path = graph_path
        self.device = torch.device("cpu")

        with open(registry_path) as f:
            self.registry = json.load(f)

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


    def predict(self):

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

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:,1]

        return {
            "count": int(probabilities.shape[0]),
            "threshold": float(self.threshold),
            "fraud_predictions":
                int((probabilities >= self.threshold).sum()),
        }
