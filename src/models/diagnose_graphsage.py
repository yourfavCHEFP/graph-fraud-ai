"""
Phase 10.5 - GraphSAGE Diagnostic Analysis.

Purpose:
    Diagnose GraphSAGE training collapse before further optimization.

Checks:
    1. Graph and mask integrity
    2. Class distribution
    3. Feature statistics
    4. Graph degree statistics
    5. Baseline GraphSAGE prediction distribution
    6. Probability separation between fraud and normal nodes
    7. Prediction/class collapse
    8. Validation vs test behaviour

Input:
    data/graph/fraud_graph_ready.pt
    models/graphsage_best.pt

Output:
    Console diagnostics only.
"""

import os

import numpy as np
import torch

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
MODEL_PATH = "models/graphsage_best.pt"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print("\n==============================")
    print("LOADING GRAPH")
    print("==============================")

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(
            f"Graph not found: {GRAPH_PATH}"
        )

    graph = torch.load(
        GRAPH_PATH,
        weights_only=False
    )

    print(graph)

    print(
        f"Nodes: {graph.x.shape[0]:,}"
    )

    print(
        f"Features: {graph.x.shape[1]}"
    )

    print(
        f"Edges: {graph.edge_index.shape[1]:,}"
    )

    return graph


# ============================================================
# MASK DIAGNOSTICS
# ============================================================

def diagnose_masks(graph):

    print("\n==============================")
    print("MASK DIAGNOSTICS")
    print("==============================")

    transaction_mask = graph.transaction_mask

    train_mask = (
        transaction_mask
        & graph.train_mask
    )

    val_mask = (
        transaction_mask
        & graph.val_mask
    )

    test_mask = (
        transaction_mask
        & graph.test_mask
    )

    print(
        f"Transaction nodes: "
        f"{transaction_mask.sum().item():,}"
    )

    print(
        f"Training transactions: "
        f"{train_mask.sum().item():,}"
    )

    print(
        f"Validation transactions: "
        f"{val_mask.sum().item():,}"
    )

    print(
        f"Test transactions: "
        f"{test_mask.sum().item():,}"
    )

    overlap_train_val = (
        train_mask & val_mask
    ).sum().item()

    overlap_train_test = (
        train_mask & test_mask
    ).sum().item()

    overlap_val_test = (
        val_mask & test_mask
    ).sum().item()

    print(
        f"\nTrain/Val overlap: "
        f"{overlap_train_val}"
    )

    print(
        f"Train/Test overlap: "
        f"{overlap_train_test}"
    )

    print(
        f"Val/Test overlap: "
        f"{overlap_val_test}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def diagnose_classes(graph):

    print("\n==============================")
    print("CLASS DISTRIBUTION")
    print("==============================")

    transaction_mask = graph.transaction_mask

    train_mask = (
        transaction_mask
        & graph.train_mask
    )

    val_mask = (
        transaction_mask
        & graph.val_mask
    )

    test_mask = (
        transaction_mask
        & graph.test_mask
    )

    for name, mask in [
        ("TRAIN", train_mask),
        ("VALIDATION", val_mask),
        ("TEST", test_mask),
    ]:

        labels = (
            graph.y[mask]
            .cpu()
            .numpy()
        )

        normal = int(
            (labels == 0).sum()
        )

        fraud = int(
            (labels == 1).sum()
        )

        total = len(labels)

        fraud_rate = (
            fraud / total
            if total > 0
            else 0
        )

        print(
            f"\n{name}"
        )

        print(
            f"Total:  {total:,}"
        )

        print(
            f"Normal: {normal:,}"
        )

        print(
            f"Fraud:  {fraud:,}"
        )

        print(
            f"Fraud rate: "
            f"{fraud_rate:.4%}"
        )


# ============================================================
# FEATURE DIAGNOSTICS
# ============================================================

def diagnose_features(graph):

    print("\n==============================")
    print("FEATURE DIAGNOSTICS")
    print("==============================")

    x = graph.x.float()

    train_mask = (
        graph.transaction_mask
        & graph.train_mask
    )

    train_x = x[train_mask]

    feature_names = [
        "degree",
        "transaction_entity_count",
        "log_transaction_amount",
        "card_degree",
        "email_degree",
        "device_degree",
        "address_degree",
        "node_type_id",
    ]

    print("\nFeature statistics:")

    for index, name in enumerate(
        feature_names
    ):

        values = (
            train_x[:, index]
            .cpu()
            .numpy()
        )

        print(
            f"\n{name}"
        )

        print(
            f"  min:    {values.min():.4f}"
        )

        print(
            f"  max:    {values.max():.4f}"
        )

        print(
            f"  mean:   {values.mean():.4f}"
        )

        print(
            f"  std:    {values.std():.4f}"
        )

        print(
            f"  median: {np.median(values):.4f}"
        )

        if not np.isfinite(values).all():

            print(
                "  WARNING: non-finite values detected!"
            )


# ============================================================
# GRAPH DEGREE DIAGNOSTICS
# ============================================================

def diagnose_graph_structure(graph):

    print("\n==============================")
    print("GRAPH STRUCTURE DIAGNOSTICS")
    print("==============================")

    edge_index = (
        graph.edge_index.cpu()
    )

    num_nodes = graph.x.shape[0]

    degrees = torch.bincount(
        edge_index[0],
        minlength=num_nodes
    )

    degrees_np = (
        degrees.numpy()
    )

    print(
        f"\nMinimum degree: "
        f"{degrees_np.min()}"
    )

    print(
        f"Maximum degree: "
        f"{degrees_np.max()}"
    )

    print(
        f"Mean degree: "
        f"{degrees_np.mean():.4f}"
    )

    print(
        f"Median degree: "
        f"{np.median(degrees_np):.4f}"
    )

    print(
        f"95th percentile: "
        f"{np.percentile(degrees_np, 95):.4f}"
    )

    print(
        f"99th percentile: "
        f"{np.percentile(degrees_np, 99):.4f}"
    )

    print("\nTop 10 node degrees:")

    top_indices = np.argsort(
        degrees_np
    )[-10:][::-1]

    for rank, node_index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"Node {node_index:,} "
            f"degree={degrees_np[node_index]:,}"
        )


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize_features(graph):

    print("\n==============================")
    print("FEATURE NORMALIZATION")
    print("==============================")

    x = graph.x.float()

    train_mask = (
        graph.transaction_mask
        & graph.train_mask
    )

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
        x - mean
    ) / std

    print(
        "Training-based normalization applied."
    )

    return graph


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(graph):

    print("\n==============================")
    print("LOADING GRAPHSAGE MODEL")
    print("==============================")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\nGraphSAGE model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run train_gnn.py first so that "
            "models/graphsage_best.pt exists."
        )

    model = FraudGraphSAGE(
        input_dim=graph.x.shape[1],
        hidden_dim=64,
        output_dim=2,
        dropout=0.3
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(DEVICE)

    model.eval()

    print(
        f"Loaded: {MODEL_PATH}"
    )

    return model


# ============================================================
# PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    graph
):

    print("\n==============================")
    print("GENERATING PREDICTIONS")
    print("==============================")

    graph = graph.to(DEVICE)

    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[:, 1]

    return (
        output,
        probabilities
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

def diagnose_predictions(
    graph,
    probabilities
):

    print("\n==============================")
    print("PREDICTION DISTRIBUTION")
    print("==============================")

    transaction_mask = (
        graph.transaction_mask
    )

    train_mask = (
        transaction_mask
        & graph.train_mask
    )

    val_mask = (
        transaction_mask
        & graph.val_mask
    )

    test_mask = (
        transaction_mask
        & graph.test_mask
    )

    for name, mask in [
        ("TRAIN", train_mask),
        ("VALIDATION", val_mask),
        ("TEST", test_mask),
    ]:

        probs = (
            probabilities[mask]
            .detach()
            .cpu()
            .numpy()
        )

        print(
            f"\n{name}"
        )

        print(
            f"min:    {probs.min():.8f}"
        )

        print(
            f"max:    {probs.max():.8f}"
        )

        print(
            f"mean:   {probs.mean():.8f}"
        )

        print(
            f"median: {np.median(probs):.8f}"
        )

        print(
            f"std:    {probs.std():.8f}"
        )

        print(
            f"p90:    {np.percentile(probs, 90):.8f}"
        )

        print(
            f"p99:    {np.percentile(probs, 99):.8f}"
        )

        print(
            f"p99.9:  {np.percentile(probs, 99.9):.8f}"
        )


# ============================================================
# CLASS SEPARATION
# ============================================================

def diagnose_class_separation(
    graph,
    probabilities
):

    print("\n==============================")
    print("FRAUD / NORMAL SCORE SEPARATION")
    print("==============================")

    mask = (
        graph.transaction_mask
        & graph.val_mask
    )

    y_true = (
        graph.y[mask]
        .detach()
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities[mask]
        .detach()
        .cpu()
        .numpy()
    )

    fraud_probs = y_prob[
        y_true == 1
    ]

    normal_probs = y_prob[
        y_true == 0
    ]

    print(
        "\nValidation fraud scores:"
    )

    print(
        f"Count:  {len(fraud_probs):,}"
    )

    print(
        f"Mean:   {fraud_probs.mean():.8f}"
    )

    print(
        f"Median: {np.median(fraud_probs):.8f}"
    )

    print(
        f"90th:   {np.percentile(fraud_probs, 90):.8f}"
    )

    print(
        f"Max:    {fraud_probs.max():.8f}"
    )

    print(
        "\nValidation normal scores:"
    )

    print(
        f"Count:  {len(normal_probs):,}"
    )

    print(
        f"Mean:   {normal_probs.mean():.8f}"
    )

    print(
        f"Median: {np.median(normal_probs):.8f}"
    )

    print(
        f"90th:   {np.percentile(normal_probs, 90):.8f}"
    )

    print(
        f"Max:    {normal_probs.max():.8f}"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def diagnose_performance(
    graph,
    probabilities
):

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")

    for name, split_mask in [
        (
            "VALIDATION",
            graph.val_mask
        ),
        (
            "TEST",
            graph.test_mask
        ),
    ]:

        mask = (
            graph.transaction_mask
            & split_mask
        )

        y_true = (
            graph.y[mask]
            .detach()
            .cpu()
            .numpy()
        )

        y_prob = (
            probabilities[mask]
            .detach()
            .cpu()
            .numpy()
        )

        roc = roc_auc_score(
            y_true,
            y_prob
        )

        pr = average_precision_score(
            y_true,
            y_prob
        )

        threshold = 0.5

        predictions = (
            y_prob >= threshold
        ).astype(int)

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

        print(
            f"\n{name}"
        )

        print(
            f"ROC-AUC:   {roc:.4f}"
        )

        print(
            f"PR-AUC:    {pr:.4f}"
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
            "\nConfusion Matrix:"
        )

        print(
            matrix
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================")
    print("PHASE 10.5 GRAPHSAGE DIAGNOSTICS")
    print("==============================")

    print(
        f"\nDevice: {DEVICE}"
    )

    graph = load_graph()

    diagnose_masks(
        graph
    )

    diagnose_classes(
        graph
    )

    diagnose_features(
        graph
    )

    diagnose_graph_structure(
        graph
    )

    graph = normalize_features(
        graph
    )

    model = load_model(
        graph
    )

    output, probabilities = (
        generate_predictions(
            model,
            graph
        )
    )

    diagnose_predictions(
        graph,
        probabilities
    )

    diagnose_class_separation(
        graph,
        probabilities
    )

    diagnose_performance(
        graph,
        probabilities
    )

    print("\n==============================")
    print("PHASE 10.5 DIAGNOSTICS COMPLETE")
    print("==============================")


if __name__ == "__main__":
    main()
