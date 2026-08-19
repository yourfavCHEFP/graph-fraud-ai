"""Production inference service for the GraphSAGE fraud detector."""

import json
import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.explainability.fraud_reasoner import build_fraud_explanation
from src.features.graph_features import build_graph_features

load_dotenv()

logger = logging.getLogger("graph-fraud-api")

_LFS_POINTER_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"


def _assert_not_lfs_pointer(path: Path) -> None:
    """
    Git LFS-tracked files that were never actually pulled (e.g. a plain
    `git clone`/`docker build` without git-lfs configured, or a `git lfs
    pull` bandwidth quota exceeded) leave a small text pointer file in
    place of the real binary -- typically ~130 bytes, starting with
    "version https://git-lfs.github.com/spec/v1". Loading that with
    torch.load() previously failed with a deep, unclear pickle error deep
    inside torch's internals, on the first live /predict request, with
    nothing useful reaching the logs. This check catches it immediately,
    at the exact file, with a message that says what's actually wrong.
    """
    with open(path, "rb") as f:
        header = f.read(len(_LFS_POINTER_SIGNATURE))
    if header == _LFS_POINTER_SIGNATURE:
        raise RuntimeError(
            f"{path} is a Git LFS POINTER FILE, not the real artifact "
            f"(this happens when the environment building/deploying the app "
            f"doesn't have git-lfs pulling the actual binary content -- 'git "
            f"clone' without LFS support leaves this tiny text stub in place "
            f"of the real file). Fix: verify Git LFS is actually fetching "
            f"real content in this deployment environment, or serve this "
            f"artifact from a direct download URL instead of relying on LFS "
            f"at deploy time."
        )


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

        _assert_not_lfs_pointer(checkpoint_path)
        _assert_not_lfs_pointer(self.graph_path)

        from src.models.gnn_model import FraudGraphSAGE

        logger.info("Loading model checkpoint from %s", checkpoint_path)
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
        logger.info("Model loaded: %s", self.model_name)

        self.mean = checkpoint["normalization_mean"]
        self.std = checkpoint["normalization_std"]

        logger.info("Loading graph artifact from %s", self.graph_path)
        self.graph = torch.load(
            self.graph_path, map_location="cpu", weights_only=False
        )
        logger.info("Graph loaded: %d nodes", self.graph.x.shape[0])

        # FIX: this forward pass used to run lazily on the FIRST live
        # /predict request -- meaning the most expensive operation in the
        # whole service (a full-graph GraphSAGE forward pass over every
        # node) happened inside the request/response cycle, where a slow
        # host or tight memory limit manifests as a silent 502 with nothing
        # in the logs (the request never completes, so uvicorn's access
        # log line for it never gets written). Running it here, in the
        # constructor -- called once from FastAPI's startup event, not
        # per-request -- means the exact same cost is paid during Render's
        # startup/deploy phase instead, where a failure or timeout shows up
        # clearly in deploy logs instead of as a mystery 502.
        logger.info("Running one-time full-graph forward pass...")
        features = build_graph_features(self.graph)
        self.features = (features - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(self.features, self.graph.edge_index)
            self.probabilities = torch.softmax(logits, dim=1)[:, 1]
        logger.info("Forward pass complete -- predictor ready to serve.")

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
            # Should be unreachable -- __init__ always computes this now.
            # Left as a defensive guard in case FraudPredictor is ever
            # constructed some other way.
            raise RuntimeError("Predictor probabilities were never computed at startup.")

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
