# Hosting Guide

## What's committed vs. what's a secret

- **Source code**: committed normally.
- **Model checkpoint and graph artifact** (`models/production/*.pt`,
  `data/graph/*.pt`): committed **via Git LFS**, intentionally -- this
  section previously said the opposite (do not commit checkpoints or
  graph artifacts); that was wrong and contradicted the repository's
  actual, deliberate LFS-based deployment strategy. Correction: DO
  commit them, through LFS, not as raw large blobs.
- **Secrets** (API keys, tokens): never committed. Use environment
  variables (`.env`, Render/Space environment settings) -- see
  `.env.example` for the full list this project uses.

## Render

Use the Dockerfile as the service image and expose port `8000`. Set
runtime environment variables from `.env.example`. The model checkpoint
and graph artifact arrive via the normal `git clone` + LFS pull as part
of the build -- see `docs/deployment/production-runbook.md` for how to
verify LFS actually resolved real content in that environment, and for
the known full-graph startup memory limitation on Render's free tier.

## Hugging Face Spaces

**Status: active**, not legacy. `deployment/huggingface/` is a real
Streamlit Space (`sdk: streamlit`, `app_file: app.py`) that runs
inference in-process -- not a static placeholder page. It is kept in
sync with the authoritative `src/` package by
`.github/workflows/sync-to-hf-space.yml` (via
`scripts/package_huggingface_space.py`) rather than a hand-maintained
duplicate copy. The model checkpoint and graph artifact are pushed
directly to the Space's own git repo via Git LFS (see
`deployment/huggingface/README.md`), since the sync workflow only
handles code, not large binaries.

## Streamlit Community Cloud

The currently-recommended primary live demo link (see main `README.md`).
Deploys `deployment/streamlit/app.py` directly from this GitHub repo;
`API_URL` points at the Render-hosted FastAPI service.

## Security

Do not commit `.env` files or raw financial data. Model checkpoints and
graph artifacts ARE committed, deliberately, through Git LFS -- see
above.
