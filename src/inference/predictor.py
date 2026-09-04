"""Production inference service for the GraphSAGE fraud detector."""

import gc
import json
import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.explainability.fraud_reasoner import build_fraud_explanation
from src.explainability.graph_explainer import build_adjacency_index
from src.features.graph_features import build_graph_features, get_feature_names

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


def _validate_checkpoint_contract(checkpoint: dict, graph_feature_dim: int) -> None:
    """
    FIX (mentor review item 6): previously nothing verified that a loaded
    checkpoint actually matches the graph it's about to run against. A
    checkpoint trained against an older/different feature-column layout
    (see the graph_features.py column-alignment bug, item 1) would load
    and run without error -- just silently produce wrong predictions.
    This raises loudly instead, on every field that's checkable.

    Fields checked unconditionally (present in every checkpoint format
    seen so far): input_dim, normalization_mean/std length.

    Fields checked only if present (forward-compatible with the retrained
    checkpoint format item 5 asks for, which adds feature_names and
    feature_pipeline_version -- older checkpoints won't have these yet,
    so their absence is logged, not fatal).
    """
    required_keys = ["input_dim", "hidden_dim", "dropout", "model_state_dict",
                      "normalization_mean", "normalization_std", "validation_threshold"]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing required key(s): {missing}")

    input_dim = checkpoint["input_dim"]

    if input_dim != graph_feature_dim:
        raise ValueError(
            f"Checkpoint was trained with input_dim={input_dim}, but the "
            f"current feature pipeline (src.features.graph_features) "
            f"produces {graph_feature_dim} features for this graph. This "
            f"checkpoint was trained against a different feature contract "
            f"than the one currently in code -- do not serve predictions "
            f"from this pairing. Retrain against the current pipeline."
        )

    mean_len = checkpoint["normalization_mean"].numel()
    std_len = checkpoint["normalization_std"].numel()
    if mean_len != input_dim or std_len != input_dim:
        raise ValueError(
            f"Checkpoint normalization_mean/std length ({mean_len}/{std_len}) "
            f"does not match input_dim ({input_dim})."
        )

    if "feature_names" in checkpoint:
        current_names = get_feature_names()
        if checkpoint["feature_names"] != current_names:
            raise ValueError(
                f"Checkpoint's recorded feature_names do not match the "
                f"current feature pipeline's output.\n"
                f"  Checkpoint : {checkpoint['feature_names']}\n"
                f"  Current    : {current_names}"
            )
    else:
        logger.warning(
            "Checkpoint has no 'feature_names' field to verify against -- "
            "it predates the feature-schema versioning this contract check "
            "expects (see mentor review item 5). Cannot confirm feature "
            "identity/order matches, only dimension count."
        )

    if "feature_pipeline_version" in checkpoint:
        logger.info("Checkpoint feature_pipeline_version: %s", checkpoint["feature_pipeline_version"])
    else:
        logger.warning("Checkpoint has no 'feature_pipeline_version' field.")


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

        logger.info("Loading graph artifact from %s", self.graph_path)
        self.graph = torch.load(
            self.graph_path, map_location="cpu", weights_only=False
        )
        logger.info("Graph loaded: %d nodes", self.graph.x.shape[0])

        # Contract check happens BEFORE building the model or running any
        # inference -- a mismatch should fail startup immediately, not
        # after paying the cost of a full-graph forward pass.
        graph_feature_dim = build_graph_features(self.graph).shape[1]
        _validate_checkpoint_contract(checkpoint, graph_feature_dim)

        # FIX (mentor review item 6): threshold ambiguity. Previously
        # DEFAULT_FRAUD_THRESHOLD silently overrode the checkpoint's own
        # validation_threshold whenever the env var was set at all --
        # render.yaml set it to 0.5 while the checkpoint's actual tuned
        # threshold was 0.26, so production quietly used a DIFFERENT
        # threshold than the one the evaluated metrics were computed at.
        # The checkpoint is now authoritative by default. Overriding it
        # requires the explicitly-named FRAUD_THRESHOLD_OVERRIDE env var
        # (not "DEFAULT_..." -- a default that silently wins isn't a
        # default, it's an override with a misleading name), and doing so
        # is logged loudly so it's never a silent discrepancy again.
        self.threshold = float(checkpoint["validation_threshold"])
        override = os.getenv("FRAUD_THRESHOLD_OVERRIDE")
        if override is not None:
            logger.warning(
                "FRAUD_THRESHOLD_OVERRIDE is set -- using %s instead of the "
                "checkpoint's own validation_threshold (%s). This means "
                "production is NOT using the threshold the evaluated "
                "metrics were computed at. Make sure that's intentional.",
                override, checkpoint["validation_threshold"],
            )
            self.threshold = float(override)

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

        self.num_nodes = self.graph.x.shape[0]
        logger.info("Building adjacency index for neighbor lookups...")
        self.adjacency_index = build_adjacency_index(self.graph)
        logger.info("Adjacency index built for %d source nodes.", len(self.adjacency_index))

        # FIX (OOM on Render free tier, 512MB limit; mentor review item 11):
        # self.features (the normalized feature matrix) and self.model
        # (weights) are NOT touched again after this point --
        # predict_transaction() below only uses
        # self.num_nodes/self.adjacency_index/self.probabilities/self.threshold.
        # self.graph itself is ALSO no longer needed now that the
        # adjacency index has been built from it -- freeing it too, on top
        # of the earlier model/features cleanup, further reduces
        # steady-state memory.
        del self.graph, self.features, self.model, features, logits
        gc.collect()
        logger.info("Released graph/model/feature tensors after forward pass + adjacency indexing.")

    @property
    def model_name(self):
        return self.registry["production_model"]["name"]

    def predict_transaction(self, transaction_id: int):
        transaction_id = int(transaction_id)
        if transaction_id < 0 or transaction_id >= self.num_nodes:
            raise ValueError(
                f"transaction_id must be between 0 and {self.num_nodes - 1}"
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
                self.adjacency_index, self.num_nodes, transaction_id, probability, self.threshold
            ),
        }
