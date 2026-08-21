---
title: Graph-Fraud AI Demo
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# 🛡️ Graph-Fraud AI: Production Demo

This Hugging Face Space hosts the production inference engine for the **Graph-Fraud AI** project. It demonstrates how a trained **GraphSAGE** model can be deployed to provide real-time fraud detection and relational explainability.

## 🏗️ The Bridge Architecture

This project follows a professional separation of concerns between research and production:

1.  **GitHub (Research & Development):** Contains the full ML pipeline, 10-notebook series, graph construction logic, and model training experiments.
2.  **Hugging Face (Production Demo):** Hosts this interactive dashboard, running the inference engine in-process for low-latency predictions and graph visualizations.

## 🚀 Features

*   **Real-time Inference:** Predicts fraud probability for any transaction in the graph using GraphSAGE.
*   **Relational Explainability:** Breaks down *why* a transaction is flagged, highlighting specific graph-based risk factors.
*   **Graph Visualization:** Renders the local neighborhood of a transaction to show its connections to other entities (cards, devices, IPs).

## 🛠️ Technical Stack

*   **Framework:** PyTorch Geometric (PyG)
*   **Model:** Heterogeneous GraphSAGE
*   **UI:** Streamlit
*   **Graph Logic:** NetworkX & Matplotlib

## 📦 Setup & Deployment Notes

The model checkpoint and graph file are large binaries and must be tracked with **Git LFS** in the Hugging Face Space repository:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/production/graphsage_improved.pt data/graph/fraud_graph_ready.pt
git commit -m "Add model + graph artifacts"
git push
```

For the full source code, research notebooks, and training pipeline, visit the [GitHub Repository](https://github.com/yourfavCHEFP/graph-fraud-ai).
