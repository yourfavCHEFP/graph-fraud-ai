"""Production inference service for the GraphSAGE fraud detector."""

import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.explainability.fraud_reasoner import build_fraud_explanation
from src.features.graph_features import build_graph_features

load_dotenv()


class FraudPredictor:
    """Load the champion model and graph once, then serve transaction predictions."""

    def __init__(
        self,
        registry_path=None,
        graph_path=None,
    ):
        self.registry_path = Path(
            registry_path or os.getenv(
                "MODEL_REGISTRY_PATH", "models/registry/model_registry.json"
            )
        )
        self.graph_path = Path(
            graph_path or os.getenv(
                "GRAPH_PATH", "data/graph/fraud_graph_ready.pt"
            )
        )

        if not self.registry_path.exists():
            raise FileNotFoundError(f"Model registry not found: {self.registry_path}")

        with self.registry_path.open() as file:
            self.registry = json.load(file)

        checkpoint_path = Path(
            os.getenv(
                "MODEL_CHECKPOINT_PATH",
                self.registry["production_model"]["checkpoint"],
            )
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph artifact not found: {self.graph_path}")

        from src.models.gnn_model import FraudGraphSAGE

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.threshold = float(
            os.getenv("DEFAULT_FRAUD_THRESHOLD", checkpoint["validation_threshold"])
        )

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

        self.graph = torch.load(
            self.graph_path, map_location="cpu", weights_only=False
        )

        # Do not execute full graph inference during startup. On Render Free,
        # doing a complete GraphSAGE forward pass during health checks can make
        # the service unavailable. Predictions are generated lazily.
        self.features = None
        self.probabilities = None

    @property
    def model_name(self):
        return self.registry["production_model"]["name"]

    def predict_transaction(self, transaction_id: int):
        transaction_id = int(transaction_id)
        if transaction_id < 0 or transaction_id >= self.graph.x.shape[0]:
            raise ValueError(
                f"transaction_id must be between 0 and {self.graph.x.shape[0] - 1}"
            )

        if self.probabilities is None:
            features = build_graph_features(self.graph)
            self.features = (features - self.mean) / self.std
            with torch.no_grad():
                logits = self.model(self.features, self.graph.edge_index)
                self.probabilities = torch.softmax(logits, dim=1)[:, 1]

        probability = float(self.probabilities[transaction_id].item())
        prediction = "fraud" if probability >= self.threshold else "legitimate"

        return {
            "transaction_id": transaction_id,
            "prediction": prediction,
            "fraud_probability": round(probability, 6),
            "threshold": round(self.threshold, 6),
            "model": "GraphSAGE",
            "explanation": build_fraud_explanation(
                self.graph, transaction_id, probability, self.threshold
            ),
        }
