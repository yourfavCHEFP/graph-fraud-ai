# Graph Fraud AI

Production-oriented Graph Neural Network fraud detection system built on the IEEE-CIS Fraud Detection dataset.

## Architecture

IEEE-CIS Dataset → Preprocessing → Graph Construction → Leakage-Safe Graph Features → PyTorch Geometric GNN → GraphSAGE Champion → Inference API → Deployment

## Current Champion Model

- Architecture: GraphSAGE
- Checkpoint: `models/graphsage_improved.pt`
- Feature pipeline: Phase 10.9
- Features: 16 leakage-safe graph features

## Performance

Test:
- ROC-AUC: 0.6760
- PR-AUC: 0.0821
- Precision: 0.1092
- Recall: 0.3775
- F1: 0.1694

## Run API

```bash
uvicorn deployment.fastapi.app:app --host 0.0.0.0 --port 8000
```

## Project Status

Phase 11: Model benchmarking and selection complete.
Phase 12: Production inference layer.
Phase 13: FastAPI serving.
Phase 14: Deployment preparation.
