"""
scripts/package_huggingface_space.py

FIX (mentor review item 29): deployment/huggingface/src/ used to be a
hand-maintained duplicate of the authoritative src/ package (models,
inference, features, explainability) -- a second copy that would
eventually drift from the real one (exactly as it had: it still had the
old buggy predictor.py with the threshold ambiguity and the pre-fix
graph_features.py bug from item 1, neither of which the fixes to the
root src/ package had reached).

This script assembles a fresh, self-contained staging directory at
DEPLOY TIME from the single authoritative src/ package, instead of
keeping a permanent hand-edited copy in git. A HF Space needs a
self-contained folder (it can't import across repo boundaries once
deployed), so *some* copy has to exist -- the fix is that it's now
generated on demand, not hand-maintained.

Usage:
    python scripts/package_huggingface_space.py --output /tmp/hf-space-staging

Then push /tmp/hf-space-staging's contents to the HF Space git remote
(see .github/workflows/sync-to-hf-space.yml, which calls this script).
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Authoritative source directories this Space needs, copied verbatim --
# no hand-editing happens on the copies, so there's nothing to drift.
REQUIRED_SRC_SUBPACKAGES = [
    "src/graph",
    "src/models",
    "src/inference",
    "src/features",
    "src/explainability",
]

# Space-specific files that are NOT duplicates of anything -- these
# genuinely only make sense for this deployment target and stay
# hand-maintained in deployment/huggingface/.
SPACE_OWNED_FILES = [
    "deployment/huggingface/app.py",
    "deployment/huggingface/requirements.txt",
    "deployment/huggingface/README.md",
]

# The model registry is a build artifact (item 24: "regenerated, not
# manually edited"), owned by models/registry/ at the repo root -- never
# by the Space folder.
REGISTRY_FILE = "models/registry/model_registry.json"


def package(output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for subpackage in REQUIRED_SRC_SUBPACKAGES:
        src = PROJECT_ROOT / subpackage
        if not src.exists():
            raise FileNotFoundError(f"Required source package missing: {src}")
        dst = output_dir / subpackage
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"Copied {subpackage}")

    # src/ needs its own __init__.py for the copied subpackages to import
    # as `src.models`, `src.inference`, etc. inside the staged Space.
    (output_dir / "src" / "__init__.py").touch()

    for space_file in SPACE_OWNED_FILES:
        src = PROJECT_ROOT / space_file
        if not src.exists():
            raise FileNotFoundError(f"Required Space file missing: {src}")
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        print(f"Copied {space_file} -> {dst.relative_to(output_dir)}")

    registry_src = PROJECT_ROOT / REGISTRY_FILE
    if registry_src.exists():
        registry_dst = output_dir / "models" / "registry" / "model_registry.json"
        registry_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(registry_src, registry_dst)
        print(f"Copied {REGISTRY_FILE}")
    else:
        print(
            f"[WARNING] {REGISTRY_FILE} not found -- the packaged Space "
            f"will fail at startup until this exists (see item 24: "
            f"regenerate the registry after retraining)."
        )

    print(f"\nHF Space package assembled at: {output_dir}")
    print(
        "NOTE: this does NOT include the model checkpoint or graph "
        "artifact (models/production/*.pt, data/graph/*.pt) -- those "
        "must be pushed to the Space repo separately via git-lfs, since "
        "they're large binaries this script intentionally doesn't copy "
        "on every packaging run."
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Directory to assemble the self-contained Space package into.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    package(args.output)
