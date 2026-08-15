"""
Phase 10.3 - GNN Model Benchmark Evaluation.

Evaluates:
- GCN
- GAT
- GraphSAGE

Input:
    data/graph/fraud_graph_ready.pt
    models/gcn_best.pt
    models/gat_best.pt
    models/graphsage_best.pt

Evaluation:
    - Transaction nodes only
    - Test split only
    - ROC-AUC
    - PR-AUC
    - Precision
    - Recall
    - F1
    - Best F1 threshold
    - Confusion matrix

Output:
    Console benchmark
"""

import os

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.gnn_model import (
    FraudGAT,
    FraudGCN,
    FraudGraphSAGE,
)

# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = "data/graph/fraud_graph_ready.pt"

MODEL_PATHS = {
    "GCN": "models/gcn_best.pt",
    "GAT": "models/gat_best.pt",
    "GraphSAGE": "models/graphsage_best.pt",
}


# ============================================================
# LOAD GRAPH
# ============================================================


def load_graph():

    print("\nLoading fraud graph...")

    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    graph = torch.load(
        GRAPH_PATH,
        weights_only=False,
    )

    print(graph)

    return graph


# ============================================================
# CREATE MODEL
# ============================================================


def create_model(
    model_name,
    input_dim,
):

    if model_name == "GCN":

        return FraudGCN(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=2,
            dropout=0.3,
        )

    if model_name == "GAT":

        return FraudGAT(
            input_dim=input_dim,
            hidden_dim=32,
            output_dim=2,
            heads=2,
            dropout=0.3,
        )

    if model_name == "GraphSAGE":

        return FraudGraphSAGE(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=2,
            dropout=0.3,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# LOAD MODEL
# ============================================================


def load_model(
    model_name,
    input_dim,
    device,
):

    model_path = MODEL_PATHS[model_name]

    print(f"\nLoading {model_name} model...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = create_model(
        model_name,
        input_dim,
    )

    # --------------------------------------------------------
    # Support both:
    #
    # 1. Raw state_dict checkpoints
    # 2. Wrapped checkpoints containing model_state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# FIND BEST F1 THRESHOLD
# ============================================================


def find_best_threshold(
    y_true,
    probabilities,
):

    best_threshold = 0.5

    best_f1 = 0.0

    thresholds = np.linspace(
        0.001,
        0.999,
        999,
    )

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:

            best_f1 = current_f1

            best_threshold = float(threshold)

    return (
        best_threshold,
        best_f1,
    )


# ============================================================
# EVALUATE MODEL
# ============================================================


def evaluate_model(
    model_name,
    graph,
    device,
):

    print("\n==============================")
    print(f"EVALUATING {model_name}")
    print("==============================")

    model = load_model(
        model_name,
        graph.x.shape[1],
        device,
    )

    print("\nGenerating predictions...")

    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index,
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )[:, 1]

    evaluation_mask = graph.transaction_mask & graph.test_mask

    y_true = graph.y[evaluation_mask].detach().cpu().numpy()

    y_prob = probabilities[evaluation_mask].detach().cpu().numpy()

    print(
        "\nTest transactions:",
        len(y_true),
    )

    print(
        "Fraud transactions:",
        int(y_true.sum()),
    )

    print(
        "Normal transactions:",
        int((y_true == 0).sum()),
    )

    # ========================================================
    # RANKING METRICS
    # ========================================================

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob,
    )

    # ========================================================
    # THRESHOLD OPTIMIZATION
    # ========================================================

    (
        best_threshold,
        best_f1,
    ) = find_best_threshold(
        y_true,
        y_prob,
    )

    best_predictions = (y_prob >= best_threshold).astype(int)

    precision = precision_score(
        y_true,
        best_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        best_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        best_predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        best_predictions,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nMetrics:")

    print(f"ROC-AUC:        {roc_auc:.4f}")

    print(f"PR-AUC:         {pr_auc:.4f}")

    print(f"Best Threshold: {best_threshold:.4f}")

    print(f"Precision:      {precision:.4f}")

    print(f"Recall:         {recall:.4f}")

    print(f"F1:             {f1:.4f}")

    print("\nConfusion Matrix:")

    print(matrix)

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# BENCHMARK
# ============================================================


def benchmark():

    print("==============================")
    print("PHASE 10.3 GNN MODEL BENCHMARK")
    print("==============================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "\nDevice:",
        device,
    )

    graph = load_graph()

    graph = graph.to(device)

    print(
        "\nTransaction nodes:",
        graph.transaction_mask.sum().item(),
    )

    print(
        "Training transactions:",
        (graph.transaction_mask & graph.train_mask).sum().item(),
    )

    print(
        "Validation transactions:",
        (graph.transaction_mask & graph.val_mask).sum().item(),
    )

    print(
        "Test transactions:",
        (graph.transaction_mask & graph.test_mask).sum().item(),
    )

    results = []

    for model_name in MODEL_PATHS:

        result = evaluate_model(
            model_name,
            graph,
            device,
        )

        results.append(result)

    # ========================================================
    # FINAL LEADERBOARD
    # ========================================================

    results.sort(
        key=lambda x: x["pr_auc"],
        reverse=True,
    )

    print("\n")
    print("==============================")
    print("GNN MODEL LEADERBOARD")
    print("==============================")

    print(
        "\n"
        f"{'Model':<12}"
        f"{'ROC-AUC':<12}"
        f"{'PR-AUC':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Threshold':<12}"
    )

    print("-" * 84)

    for result in results:

        print(
            f"{result['model']:<12}"
            f"{result['roc_auc']:<12.4f}"
            f"{result['pr_auc']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['threshold']:<12.4f}"
        )

    winner = results[0]

    print("\n==============================")

    print(
        "BEST GNN MODEL:",
        winner["model"],
    )

    print(f"Best PR-AUC: " f"{winner['pr_auc']:.4f}")

    print(f"Best F1:     " f"{winner['f1']:.4f}")

    print(f"Threshold:   " f"{winner['threshold']:.4f}")

    print("==============================")

    print("\nPHASE 10.3 GNN BENCHMARK COMPLETE")


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    benchmark()
