# Hosting Guide

## Render
Use the Dockerfile as the service image and expose port `8000`. Set runtime environment variables from `.env.example`. The model checkpoint and graph artifact must be mounted or supplied through the deployment artifact strategy.

## Hugging Face Spaces
Use `deployment/huggingface/` for the public demo layer. Keep secrets out of the repository and point the demo at the hosted inference endpoint when the model artifacts are not suitable for the Space.

## Security
Do not commit `.env`, raw financial data, model checkpoints, or graph artifacts.
