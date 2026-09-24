"""Runtime settings shared by inference and deployment checks."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    model_registry_path: Path
    graph_path: Path
    model_checkpoint_path: Path | None
    fraud_threshold_override: float | None

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        root_dir = (base_dir or PROJECT_ROOT).resolve()

        def configured_path(name: str, default: str) -> Path:
            path = Path(os.getenv(name, default))
            return path if path.is_absolute() else root_dir / path

        threshold = os.getenv("FRAUD_THRESHOLD_OVERRIDE")
        threshold_override = None
        if threshold is not None:
            try:
                threshold_override = float(threshold)
            except ValueError as exc:
                raise ValueError("FRAUD_THRESHOLD_OVERRIDE must be a number") from exc

        checkpoint = os.getenv("MODEL_CHECKPOINT_PATH")
        return cls(
            root_dir=root_dir,
            model_registry_path=configured_path(
                "MODEL_REGISTRY_PATH", "models/registry/model_registry.json"
            ),
            graph_path=configured_path("GRAPH_PATH", "data/graph/fraud_graph_ready.pt"),
            model_checkpoint_path=(
                configured_path("MODEL_CHECKPOINT_PATH", checkpoint) if checkpoint else None
            ),
            fraud_threshold_override=threshold_override,
        )

    def resolve_path(self, path: Path | str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root_dir / path
