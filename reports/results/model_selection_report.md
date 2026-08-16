# Model Selection Report

## Project

Graph Fraud AI

## Phase

11.2 - Champion Model Selection

---

# Objective

Select the best-performing GNN architecture for financial fraud detection using benchmark results and previous optimized training results.

---

# Candidate Models

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| GCN | 0.5470 | 0.0392 |
| GAT | 0.4821 | 0.0370 |
| GraphSAGE Benchmark | 0.6615 | 0.0763 |
| GraphSAGE Improved | 0.6760 | 0.0821 |

---

# Selected Champion Model

## GraphSAGE Improved

Checkpoint:
models/graphsage_improved.pt


---

# Selection Reason

GraphSAGE achieved the strongest performance on the fraud detection task.

The optimized GraphSAGE model achieved:

- ROC-AUC: 0.6760
- PR-AUC: 0.0821
- Recall: 0.3775
- F1 Score: 0.1694

Because fraud detection contains severe class imbalance, PR-AUC and recall were prioritized over accuracy.

---

# Final Architecture Decision

The production architecture will use:
IEEE-CIS Dataset
    ↓
Graph Construction
    ↓
Graph Feature Engineering
    ↓
GraphSAGE GNN
    ↓
Inference Engine
    ↓
API Deployment

---

# Deployment Status

Current status:

Training complete.

Deployment pending.

Next phase:

Phase 12 - Inference Engine Development.