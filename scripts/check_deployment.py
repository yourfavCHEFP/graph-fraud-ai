"""Validate production deployment prerequisites without requiring model artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    ".env.example",
    "deployment/fastapi/app.py",
    "deployment/fastapi/routes/predict.py",
    "src/inference/predictor.py",
    "models/registry/model_registry.json",
]


def main():
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    registry = json.loads((ROOT / "models/registry/model_registry.json").read_text())
    checkpoint = ROOT / registry["production_model"]["checkpoint"]
    graph = ROOT / "data/graph/fraud_graph_ready.pt"

    print("Graph Fraud AI deployment audit")
    print("=" * 40)
    print(f"Required source files: {'OK' if not missing else 'MISSING'}")
    if missing:
        for item in missing:
            print(f"  - {item}")
    print(f"Champion: {registry['production_model']['name']}")
    print(f"Checkpoint artifact: {'PRESENT' if checkpoint.exists() else 'MISSING'}")
    print(f"Graph artifact: {'PRESENT' if graph.exists() else 'MISSING'}")

    if missing:
        raise SystemExit(1)

    print("Source-level deployment audit passed.")


if __name__ == "__main__":
    main()
