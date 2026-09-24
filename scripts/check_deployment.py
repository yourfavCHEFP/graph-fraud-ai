"""Validate production deployment prerequisites.

FIX (mentor review item 34): this used to only exit non-zero when a
required SOURCE FILE was missing -- a missing or LFS-pointer artifact
(checkpoint/graph) was reported in the printout but never caused a
non-zero exit, so this script could report "audit passed" in CI/a
deploy hook while the actual model artifacts were unusable. It now
distinguishes and fails on all four states: missing source file,
missing artifact, artifact present but an LFS pointer, artifact present
and readable. It also reads the checkpoint/graph paths the SAME way the
running service does (registry + env vars), not a separate hardcoded
path that could silently drift from what's actually served.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.inference.contracts import RegistryDocument  # noqa: E402

REQUIRED_SOURCE_FILES = [
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

_LFS_POINTER_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"


def _artifact_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    with open(path, "rb") as f:
        header = f.read(len(_LFS_POINTER_SIGNATURE))
    if header == _LFS_POINTER_SIGNATURE:
        return "LFS_POINTER"
    return "PRESENT"


def main():
    ok = True

    print("Graph Fraud AI deployment audit")
    print("=" * 40)

    missing_source = [p for p in REQUIRED_SOURCE_FILES if not (ROOT / p).exists()]
    print(f"Required source files: {'OK' if not missing_source else 'MISSING'}")
    if missing_source:
        ok = False
        for item in missing_source:
            print(f"  - {item}")

    settings = Settings.from_env(base_dir=ROOT)
    registry_path = settings.model_registry_path
    if not registry_path.exists():
        print("Registry: MISSING -- cannot check artifact paths.")
        raise SystemExit(1)

    registry = RegistryDocument.model_validate_json(registry_path.read_text())

    # Same resolution order as src/inference/predictor.py.
    checkpoint_path = settings.model_checkpoint_path or settings.resolve_path(
        registry.production_model.checkpoint
    )

    graph_path = settings.graph_path

    print(f"Champion: {registry.production_model.name}")

    checkpoint_status = _artifact_status(checkpoint_path)
    graph_status = _artifact_status(graph_path)

    print(f"Checkpoint artifact ({checkpoint_path}): {checkpoint_status}")
    print(f"Graph artifact ({graph_path}): {graph_status}")

    for label, status in [("Checkpoint", checkpoint_status), ("Graph", graph_status)]:
        if status == "MISSING":
            print(f"  [FAIL] {label} artifact is missing.")
            ok = False
        elif status == "LFS_POINTER":
            print(
                f"  [FAIL] {label} artifact is a Git LFS POINTER FILE, not the real "
                f"binary -- git-lfs did not fetch real content in this environment."
            )
            ok = False

    if registry.deployment_ready is False:
        print(
            "  [WARNING] Registry declares deployment_ready=false: "
            f"{registry.deployment_note or '(no note)'}"
        )

    if not ok:
        print("\nDeployment audit FAILED.")
        raise SystemExit(1)

    print("\nDeployment audit passed.")


if __name__ == "__main__":
    main()
