"""
Phase 10.4 - GraphSAGE Hyperparameter Optimization.

Uses validation PR-AUC to select the best GraphSAGE configuration.

Input:
    data/graph/fraud_graph_ready.pt

Output:
    models/graphsage_optimized.pt
"""

import os

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.gnn_model import FraudGraphSAGE


GRAPH_PATH = "data/graph/fraud_graph_ready.pt"
MODEL_PATH = "models/graphsage_optimized.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


CONFIGURATIONS = [

    {
        "hidden_dim": 32,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
    },

    {
        "hidden_dim": 64,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
    },

    {
        "hidden_dim": 128,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
    },

    {
        "hidden_dim": 64,
        "dropout": 0.5,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
    },

    {
        "hidden_dim": 64,
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "weight_decay": 0.001,
    },

]


EPOCHS = 50
FRAUD_CLASS_WEIGHT = 5.0


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print("\nLoading fraud graph...")

    graph = torch.load(
        GRAPH_PATH,
        weights_only=False
    )

    print(graph)

    return graph


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize_features(graph):

    print("\nNormalizing node features...")

    x = graph.x.float()

    train_mask = graph.train_mask

    train_x = x[train_mask]

    mean = train_x.mean(
        dim=0,
        keepdim=True
    )

    std = train_x.std(
        dim=0,
        keepdim=True
    )

    std[std == 0] = 1.0

    graph.x = (
        (x - mean) / std
    )

    return graph


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    input_dim,
    config
):

    model = FraudGraphSAGE(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        output_dim=2,
        dropout=config["dropout"],
    )

    return model.to(DEVICE)


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation(
    model,
    graph
):

    model.eval()

    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[:, 1]

    mask = (
        graph.transaction_mask
        &
        graph.val_mask
    )

    y_true = (
        graph.y[mask]
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities[mask]
        .cpu()
        .numpy()
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    return (
        pr_auc,
        roc_auc,
        y_true,
        y_prob
    )


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.linspace(
        0.001,
        0.999,
        999
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = threshold

    return (
        best_threshold,
        best_f1
    )


# ============================================================
# TRAIN ONE CONFIGURATION
# ============================================================

def train_configuration(
    graph,
    config,
    configuration_number
):

    print("\n==============================")
    print(
        f"CONFIGURATION "
        f"{configuration_number}/{len(CONFIGURATIONS)}"
    )
    print("==============================")

    print(config)

    model = create_model(
        graph.x.shape[1],
        config
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    class_weights = torch.tensor(
        [
            1.0,
            FRAUD_CLASS_WEIGHT
        ],
        dtype=torch.float,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    train_mask = (
        graph.transaction_mask
        &
        graph.train_mask
    )

    best_pr_auc = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        output = model(
            graph.x,
            graph.edge_index
        )

        loss = criterion(
            output[train_mask],
            graph.y[train_mask]
        )

        loss.backward()

        optimizer.step()

        (
            val_pr_auc,
            val_roc_auc,
            _,
            _
        ) = evaluate_validation(
            model,
            graph
        )

        if val_pr_auc > best_pr_auc:

            best_pr_auc = val_pr_auc
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch % 5 == 0
            or epoch == EPOCHS - 1
        ):

            print(
                f"Epoch {epoch:02d} | "
                f"Loss {loss.item():.4f} | "
                f"Val ROC {val_roc_auc:.4f} | "
                f"Val PR {val_pr_auc:.4f}"
            )

    print(
        f"\nBest validation PR-AUC: "
        f"{best_pr_auc:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_pr_auc,
        best_epoch
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_test(
    model,
    graph,
    threshold
):

    model.eval()

    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[:, 1]

    mask = (
        graph.transaction_mask
        &
        graph.test_mask
    )

    y_true = (
        graph.y[mask]
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities[mask]
        .cpu()
        .numpy()
    )

    predictions = (
        y_prob >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        predictions
    )

    print("\n==============================")
    print("OPTIMIZED GRAPHSAGE TEST RESULTS")
    print("==============================")

    print(
        f"ROC-AUC:  {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:   {pr_auc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"Threshold: {threshold:.4f}"
    )

    print("\nConfusion Matrix:")

    print(matrix)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================")
    print("PHASE 10.4 GRAPHSAGE OPTIMIZATION")
    print("==============================")

    print(
        f"\nDevice: {DEVICE}"
    )

    graph = load_graph()

    graph = graph.to(
        DEVICE
    )

    graph = normalize_features(
        graph
    )

    print(
        "\nFeature dimension:",
        graph.x.shape[1]
    )

    print(
        "Training transactions:",
        (
            graph.transaction_mask
            & graph.train_mask
        ).sum().item()
    )

    print(
        "Validation transactions:",
        (
            graph.transaction_mask
            & graph.val_mask
        ).sum().item()
    )

    print(
        "Test transactions:",
        (
            graph.transaction_mask
            & graph.test_mask
        ).sum().item()
    )

    results = []

    best_model = None
    best_config = None
    best_pr_auc = -1.0
    best_epoch = None

    # --------------------------------------------------------
    # Hyperparameter search
    # --------------------------------------------------------

    for index, config in enumerate(
        CONFIGURATIONS,
        start=1
    ):

        (
            model,
            validation_pr_auc,
            epoch
        ) = train_configuration(
            graph,
            config,
            index
        )

        results.append({
            "config": config,
            "validation_pr_auc": validation_pr_auc,
            "best_epoch": epoch
        })

        if validation_pr_auc > best_pr_auc:

            best_pr_auc = validation_pr_auc
            best_model = model
            best_config = config
            best_epoch = epoch

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    print("\n==============================")
    print("BEST GRAPHSAGE CONFIGURATION")
    print("==============================")

    print(
        f"Configuration: {best_config}"
    )

    print(
        f"Best validation PR-AUC: "
        f"{best_pr_auc:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    (
        _,
        _,
        validation_y,
        validation_probabilities
    ) = evaluate_validation(
        best_model,
        graph
    )

    (
        validation_threshold,
        validation_f1
    ) = find_best_threshold(
        validation_y,
        validation_probabilities
    )

    print(
        f"\nValidation threshold: "
        f"{validation_threshold:.4f}"
    )

    print(
        f"Validation F1: "
        f"{validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_results = evaluate_test(
        best_model,
        graph,
        validation_threshold
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    torch.save(
        best_model.state_dict(),
        MODEL_PATH
    )

    print(
        f"\nSaved optimized model: "
        f"{MODEL_PATH}"
    )

    print("\n==============================")
    print("PHASE 10.4 COMPLETE")
    print("==============================")


if __name__ == "__main__":

    main()
