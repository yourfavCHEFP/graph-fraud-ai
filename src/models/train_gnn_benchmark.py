"""
Phase 11.1 - Unified GNN Benchmark Training.

Purpose:
    Fair comparison of GCN, GAT, and GraphSAGE
    using identical graph features and evaluation pipeline.

Models:
    - FraudGCN
    - FraudGAT
    - FraudGraphSAGE

Evaluation:
    - ROC-AUC
    - PR-AUC
    - Precision
    - Recall
    - F1

Feature pipeline:
    16 leakage-safe graph features.

No fraud labels are used during feature construction.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from src.features.graph_features import (
    build_graph_features,
    get_feature_names,
)
from src.models.gnn_model import (
    FraudGAT,
    FraudGCN,
    FraudGraphSAGE,
)

# ============================================================
# CONFIG
# ============================================================


GRAPH_PATH = "data/graph/fraud_graph_ready.pt"

CHECKPOINT_DIR = "models/checkpoints"

REPORT_PATH = "reports/gnn_benchmark_results.csv"


SEED = 42

HIDDEN_DIM = 64

DROPOUT = 0.3

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0005

EPOCHS = 60

FRAUD_CLASS_WEIGHT = 5.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD GRAPH
# ============================================================


def load_graph():

    print("\n==============================")
    print("LOADING GRAPH")
    print("==============================")

    graph = torch.load(
        GRAPH_PATH,
        weights_only=False,
    )

    print(graph)

    return graph


# ============================================================
# FEATURE NORMALIZATION
# ============================================================


def prepare_features(graph):

    print("\n==============================")
    print("PREPARING GRAPH FEATURES")
    print("==============================")

    features = build_graph_features(graph)

    feature_names = get_feature_names()

    if features.shape[1] != len(feature_names):

        raise ValueError("Feature mismatch detected.")

    train_mask = graph.transaction_mask & graph.train_mask

    train_features = features[train_mask]

    mean = train_features.mean(
        dim=0,
        keepdim=True,
    )

    std = train_features.std(
        dim=0,
        keepdim=True,
        unbiased=False,
    )

    std[std == 0] = 1.0

    features = (features - mean) / std

    graph.x = features

    print(
        "Feature dimension:",
        graph.x.shape[1],
    )

    return graph


# ============================================================
# MODEL FACTORY
# ============================================================


def create_models(input_dim):

    return {
        "GCN": FraudGCN(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        ),
        "GAT": FraudGAT(
            input_dim=input_dim,
            hidden_dim=32,
            heads=2,
            dropout=DROPOUT,
        ),
        "GraphSAGE": FraudGraphSAGE(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        ),
    }


# ============================================================
# EVALUATION
# ============================================================


def evaluate(
    model,
    graph,
    mask,
):

    model.eval()

    with torch.no_grad():

        logits = model(
            graph.x,
            graph.edge_index,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

    y_true = graph.y[mask].cpu().numpy()

    y_prob = probabilities[mask].cpu().numpy()

    threshold = 0.5

    predictions = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


# ============================================================
# TRAIN SINGLE MODEL
# ============================================================


def train_model(
    name,
    model,
    graph,
):

    print("\n==============================")
    print(f"TRAINING {name}")
    print("==============================")

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    weights = torch.tensor(
        [
            1.0,
            FRAUD_CLASS_WEIGHT,
        ],
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(weight=weights)

    train_mask = graph.transaction_mask & graph.train_mask

    val_mask = graph.transaction_mask & graph.val_mask

    best_pr = -1

    best_state = None

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        output = model(
            graph.x,
            graph.edge_index,
        )

        loss = criterion(
            output[train_mask],
            graph.y[train_mask],
        )

        loss.backward()

        optimizer.step()

        metrics = evaluate(
            model,
            graph,
            val_mask,
        )

        if metrics["pr_auc"] > best_pr:

            best_pr = metrics["pr_auc"]

            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} "
                f"Loss {loss.item():.4f} "
                f"Val PR {metrics['pr_auc']:.4f}"
            )

    model.load_state_dict(best_state)

    test_mask = graph.transaction_mask & graph.test_mask

    results = evaluate(
        model,
        graph,
        test_mask,
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    checkpoint_path = f"{CHECKPOINT_DIR}/" f"{name.lower()}_benchmark.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": graph.x.shape[1],
            "model": name,
        },
        checkpoint_path,
    )

    results["model"] = name

    return results


# ============================================================
# MAIN
# ============================================================


def main():

    print("==============================")
    print("PHASE 11.1 GNN BENCHMARK")
    print("==============================")

    set_seed()

    print(f"\nDevice: {DEVICE}")

    print(f"Seed: {SEED}")

    # --------------------------------------------------------
    # LOAD GRAPH
    # --------------------------------------------------------

    graph = load_graph()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    graph = prepare_features(graph)

    graph = graph.to(DEVICE)

    input_dim = graph.x.shape[1]

    print(
        "\nInput dimension:",
        input_dim,
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = create_models(input_dim)

    results = []

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    for name, model in models.items():

        result = train_model(
            name,
            model,
            graph,
        )

        results.append(result)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    dataframe = pd.DataFrame(results)

    dataframe = dataframe[
        [
            "model",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
        ]
    ]

    dataframe = dataframe.sort_values(
        by="pr_auc",
        ascending=False,
    )

    dataframe.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\n==============================")
    print("BENCHMARK RESULTS")
    print("==============================")

    print(dataframe.to_string(index=False))

    print(
        "\nSaved benchmark report:",
        REPORT_PATH,
    )

    print("\n==============================")
    print("PHASE 11.1 COMPLETE")
    print("==============================")


if __name__ == "__main__":

    main()
