"""
Phase 10.4
GCN Optimization and Threshold Calibration.

Baseline:
- ROC-AUC: 0.5669
- PR-AUC: 0.0463
- F1: 0.1070

This script:
1. Loads the prepared fraud graph.
2. Normalizes node features.
3. Trains several controlled GCN configurations.
4. Selects the best configuration using validation PR-AUC.
5. Finds the best F1 threshold on validation data.
6. Evaluates the selected model on the test set.
7. Saves the best optimized GCN.

Output:
- models/gcn_optimized.pt
"""

import copy

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.models.gnn_model import FraudGCN


GRAPH_PATH = "data/graph/fraud_graph_ready.pt"
OUTPUT_PATH = "models/gcn_optimized.pt"


def load_graph():

    print("\nLoading fraud graph...")

    graph = torch.load(
        GRAPH_PATH,
        weights_only=False
    )

    print(graph)

    return graph


def normalize_features(graph):

    print("\nNormalizing node features...")

    mean = graph.x.mean(
        dim=0,
        keepdim=True
    )

    std = graph.x.std(
        dim=0,
        keepdim=True
    )

    std = torch.where(
        std == 0,
        torch.ones_like(std),
        std
    )

    graph.x = (
        graph.x - mean
    ) / std

    return graph


def calculate_class_weights(
    graph,
    device
):

    transaction_labels = graph.y[
        graph.transaction_mask
    ]

    fraud_count = (
        transaction_labels == 1
    ).sum().float()

    normal_count = (
        transaction_labels == 0
    ).sum().float()

    ratio = (
        normal_count /
        fraud_count
    )

    # Controlled weighting.
    weight = torch.sqrt(ratio)

    weight = torch.clamp(
        weight,
        max=5.0
    )

    print(
        "\nFraud ratio:",
        ratio.item()
    )

    print(
        "Fraud class weight:",
        weight.item()
    )

    return torch.tensor(
        [
            1.0,
            weight.item()
        ],
        dtype=torch.float,
        device=device
    )


def get_masks(graph):

    train_mask = (
        graph.transaction_mask
        &
        graph.train_mask
    )

    val_mask = (
        graph.transaction_mask
        &
        graph.val_mask
    )

    test_mask = (
        graph.transaction_mask
        &
        graph.test_mask
    )

    return (
        train_mask,
        val_mask,
        test_mask
    )


def get_probabilities(
    model,
    graph,
    mask,
    device
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

    return probabilities[
        mask
    ].detach().cpu().numpy()


def get_labels(
    graph,
    mask
):

    return graph.y[
        mask
    ].detach().cpu().numpy()


def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = [
        i / 1000
        for i in range(1, 1000)
    ]

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        best_threshold,
        best_f1
    )


def evaluate_validation(
    model,
    graph,
    val_mask,
    device
):

    probabilities = get_probabilities(
        model,
        graph,
        val_mask,
        device
    )

    labels = get_labels(
        graph,
        val_mask
    )

    roc = roc_auc_score(
        labels,
        probabilities
    )

    pr = average_precision_score(
        labels,
        probabilities
    )

    return (
        roc,
        pr,
        probabilities,
        labels
    )


def train_configuration(
    graph,
    device,
    hidden_dim,
    dropout,
    learning_rate,
    weight_decay,
    epochs,
    class_weights,
    train_mask,
    val_mask
):

    model = FraudGCN(
        input_dim=graph.x.shape[1],
        hidden_dim=hidden_dim,
        output_dim=2,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_pr = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(
        epochs
    ):

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

        if (
            epoch % 5 == 0
            or epoch == epochs - 1
        ):

            (
                val_roc,
                val_pr,
                _,
                _
            ) = evaluate_validation(
                model,
                graph,
                val_mask,
                device
            )

            print(
                f"Epoch {epoch:02d} | "
                f"Loss {loss.item():.4f} | "
                f"Val ROC {val_roc:.4f} | "
                f"Val PR {val_pr:.4f}"
            )

            if val_pr > best_pr:

                best_pr = val_pr

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                best_epoch = epoch

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_pr,
        best_epoch
    )


def evaluate_test(
    model,
    graph,
    test_mask,
    device,
    threshold
):

    probabilities = get_probabilities(
        model,
        graph,
        test_mask,
        device
    )

    labels = get_labels(
        graph,
        test_mask
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc = roc_auc_score(
        labels,
        probabilities
    )

    pr = average_precision_score(
        labels,
        probabilities
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        labels,
        predictions
    )

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": matrix
    }


def main():

    print("==============================")
    print("PHASE 10.4 GCN OPTIMIZATION")
    print("==============================")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    graph = load_graph()

    graph = normalize_features(
        graph
    )

    graph = graph.to(device)

    (
        train_mask,
        val_mask,
        test_mask
    ) = get_masks(graph)

    class_weights = calculate_class_weights(
        graph,
        device
    )

    configurations = [

        {
            "hidden_dim": 32,
            "dropout": 0.30,
            "learning_rate": 0.001,
            "weight_decay": 5e-4
        },

        {
            "hidden_dim": 64,
            "dropout": 0.30,
            "learning_rate": 0.001,
            "weight_decay": 5e-4
        },

        {
            "hidden_dim": 128,
            "dropout": 0.30,
            "learning_rate": 0.001,
            "weight_decay": 5e-4
        },

        {
            "hidden_dim": 64,
            "dropout": 0.50,
            "learning_rate": 0.001,
            "weight_decay": 5e-4
        },

        {
            "hidden_dim": 64,
            "dropout": 0.30,
            "learning_rate": 0.0005,
            "weight_decay": 1e-3
        }

    ]

    best_model = None
    best_config = None
    best_val_pr = -1.0
    best_epoch = None

    for index, config in enumerate(
        configurations,
        start=1
    ):

        print("\n==============================")
        print(
            f"CONFIGURATION {index}/"
            f"{len(configurations)}"
        )
        print("==============================")

        print(config)

        (
            model,
            val_pr,
            epoch
        ) = train_configuration(
            graph=graph,
            device=device,
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            epochs=50,
            class_weights=class_weights,
            train_mask=train_mask,
            val_mask=val_mask
        )

        print(
            f"\nBest validation PR-AUC: "
            f"{val_pr:.4f}"
        )

        if val_pr > best_val_pr:

            best_val_pr = val_pr
            best_model = model
            best_config = config
            best_epoch = epoch

    print("\n==============================")
    print("BEST GCN CONFIGURATION")
    print("==============================")

    print(
        "Configuration:",
        best_config
    )

    print(
        "Best validation PR-AUC:",
        f"{best_val_pr:.4f}"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    # --------------------------------------------------------
    # Threshold calibration on validation set
    # --------------------------------------------------------

    val_probabilities = get_probabilities(
        best_model,
        graph,
        val_mask,
        device
    )

    val_labels = get_labels(
        graph,
        val_mask
    )

    (
        threshold,
        validation_f1
    ) = find_best_threshold(
        val_labels,
        val_probabilities
    )

    print(
        "\nValidation threshold:",
        f"{threshold:.4f}"
    )

    print(
        "Validation F1:",
        f"{validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    results = evaluate_test(
        best_model,
        graph,
        test_mask,
        device,
        threshold
    )

    print("\n==============================")
    print("OPTIMIZED GCN TEST RESULTS")
    print("==============================")

    print(
        f"ROC-AUC:  {results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:   {results['pr_auc']:.4f}"
    )

    print(
        f"Precision: {results['precision']:.4f}"
    )

    print(
        f"Recall:    {results['recall']:.4f}"
    )

    print(
        f"F1:        {results['f1']:.4f}"
    )

    print(
        f"Threshold: {results['threshold']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        results["confusion_matrix"]
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        best_model.state_dict(),
        OUTPUT_PATH
    )

    print(
        "\nSaved optimized model:",
        OUTPUT_PATH
    )

    print("==============================")
    print("PHASE 10.4 COMPLETE")
    print("==============================")


if __name__ == "__main__":

    main()
