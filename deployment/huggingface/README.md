---
title: Graph Fraud AI
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# Graph Fraud AI

GraphSAGE-based fraud detection over a heterogeneous transaction graph
(IEEE-CIS Fraud Detection dataset). Runs entirely in this Space --
model and graph are loaded once at startup, no external API dependency.

FIX (mentor review item 27): this file previously declared `sdk: static`
with `app_file: index.html` and described the Space as a decommissioned
"static landing page" redirecting to Streamlit Cloud -- but the folder
has always contained a real Streamlit `app.py`, no `index.html` ever
existed here, and that metadata mismatch meant Hugging Face would have
failed to build this Space correctly. This is a genuine, working
Streamlit Space, not a legacy placeholder.

## How this stays in sync with the main repo

This folder's `app.py`/`requirements.txt`/`README.md` are hand-maintained
here, but the `src/` code the app imports is NOT duplicated by hand
(that used to be the case and had already drifted from real fixes made
to the authoritative package -- see mentor review item 29). Instead,
`.github/workflows/sync-to-hf-space.yml` assembles a fresh, self-contained
package from the authoritative `src/` on every relevant push and pushes
*that* to this Space's actual git repo on Hugging Face -- see
`scripts/package_huggingface_space.py`.

## Model + graph artifacts

The model checkpoint and graph file are large binaries, tracked with Git
LFS **in this Space's own git repo on Hugging Face** (not synced by the
GitHub Action above -- those only push code, since large binaries
shouldn't be re-uploaded on every code change):

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/production/graphsage_improved.pt data/graph/fraud_graph_ready.pt
git commit -m "add model + graph artifacts"
git push
```

Verify they're real (not LFS pointers) by checking their file size in
the Space's file browser -- several MB/tens of MB, not ~130 bytes.

## Environment variables (optional -- Space Settings -> Variables)

- `MODEL_REGISTRY_PATH` (default: `models/registry/model_registry.json`)
- `MODEL_CHECKPOINT_PATH` (default: from the registry)
- `GRAPH_PATH` (default: `data/graph/fraud_graph_ready.pt`)
- `FRAUD_THRESHOLD_OVERRIDE` (optional -- leave unset; the checkpoint's
  own tuned `validation_threshold` is authoritative by default)

## Also live at

Streamlit Community Cloud (`deployment/streamlit/`) hosts a version of
this demo that calls a separate FastAPI backend over HTTP. This Space
runs inference in-process instead, with no backend dependency -- see the
main repo's `README.md` for which one is the currently-recommended live
link.
