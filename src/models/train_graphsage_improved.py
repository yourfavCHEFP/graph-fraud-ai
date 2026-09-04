"""
Phase 10.6 - Improved GraphSAGE Training.

GraphSAGE fraud detector using leakage-safe graph features.

Early stopping:
    Validation PR-AUC is monitored.
    The best validation checkpoint is retained.
    Training stops after PATIENCE consecutive epochs
    without improvement.

The checkpoint also stores:
    - normalization mean
    - normalization std
    - feature names
    - input dimension
    - validation threshold
    - best validation metrics
    - model configuration
"""

import os
import random

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
from torch import nn

from src.features.graph_features import (
    build_graph_features,
    get_feature_names,
)
from src.models.gnn_model import (
    FraudGraphSAGE,
)

# ============================================================
# PATHS
# ============================================================

GRAPH_PATH = "data/graph/fraud_graph_ready.pt"

MODEL_DIR = "models"

MODEL_PATH = "models/production/graphsage_improved.pt"

FEATURE_PIPELINE_VERSION = "2026-08-31-column-alignment-fix"

# HONEST LIMITATION: this graph's train/val/test masks are the ORIGINAL
# random-stratified split (see prepare_gnn_dataset.py's chronological
# split fix -- that fix requires TransactionDT per node, which is not
# present in this graph artifact). Retraining here fixes the item-1
# feature-column bug (a real, severe issue -- see
# tests/unit/test_graph_features.py), but does NOT yet fix the item-4
# temporal-leakage issue. Both facts are recorded in the checkpoint
# below so nothing downstream has to guess or re-derive this.
SPLIT_STRATEGY = "random_stratified_LEGACY_not_chronological"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

HIDDEN_DIM = 64

DROPOUT = 0.3

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0005

EPOCHS = 100

PATIENCE = 15

FRAUD_CLASS_WEIGHT = 5.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD GRAPH
# ============================================================


def load_graph():

    print("\n==============================")
    print("LOADING FRAUD GRAPH")
    print("==============================")

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    graph = torch.load(
        GRAPH_PATH,
        weights_only=False,
    )

    print(graph)

    print(f"Nodes: {graph.x.shape[0]:,}")

    print(f"Edges: {graph.edge_index.shape[1]:,}")

    print(f"Original features: {graph.x.shape[1]}")

    return graph


# ============================================================
# NORMALIZE FEATURES
# ============================================================


def normalize_features(
    graph,
    features,
):

    print("\n==============================")
    print("TRAINING-ONLY FEATURE NORMALIZATION")
    print("==============================")

    train_mask = graph.transaction_mask & graph.train_mask

    train_features = features[train_mask]

    if train_features.numel() == 0:

        raise ValueError("No training transaction features found.")

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

    normalized_features = (features - mean) / std

    print("Normalization statistics calculated " "from training transactions only.")

    return (
        normalized_features,
        mean,
        std,
    )


# ============================================================
# CREATE MODEL
# ============================================================


def create_model(input_dim):

    model = FraudGraphSAGE(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=2,
        dropout=DROPOUT,
    )

    return model.to(DEVICE)


# ============================================================
# EVALUATE SPLIT
# ============================================================


def evaluate_split(
    model,
    graph,
    mask,
):

    model.eval()

    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index,
        )

        probability = torch.softmax(
            output,
            dim=1,
        )[:, 1]

    y_true = graph.y[mask].detach().cpu().numpy()

    y_prob = probability[mask].detach().cpu().numpy()

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob,
    )

    return (
        roc_auc,
        pr_auc,
        y_true,
        y_prob,
    )


# ============================================================
# FIND BEST THRESHOLD
# ============================================================


def find_best_threshold(
    y_true,
    y_prob,
):

    best_threshold = 0.5

    best_f1 = 0.0

    thresholds = np.linspace(
        0.001,
        0.999,
        999,
    )

    for threshold in thresholds:

        predictions = (y_prob >= threshold).astype(int)

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
# MAIN
# ============================================================


def main():

    print("==============================")
    print("PHASE 10.6 IMPROVED GRAPHSAGE")
    print("==============================")

    set_seed(SEED)

    print(f"\nRandom seed: {SEED}")

    print(f"Device: {DEVICE}")

    print(f"Hidden dimension: {HIDDEN_DIM}")

    print(f"Dropout: {DROPOUT}")

    print(f"Learning rate: {LEARNING_RATE}")

    print(f"Weight decay: {WEIGHT_DECAY}")

    print(f"Epochs: {EPOCHS}")

    print(f"Patience: {PATIENCE}")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    graph = load_graph()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    enhanced_features = build_graph_features(graph)

    feature_names = get_feature_names()

    if enhanced_features.shape[1] != len(feature_names):

        raise ValueError(
            "Feature-name mismatch: "
            f"tensor contains "
            f"{enhanced_features.shape[1]} features "
            f"but feature mapping contains "
            f"{len(feature_names)} names."
        )

    print(f"Enhanced feature dimension: " f"{enhanced_features.shape[1]}")

    print("Feature names:")

    for index, name in enumerate(feature_names):

        print(f"{index:02d}: {name}")

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    (
        normalized_features,
        normalization_mean,
        normalization_std,
    ) = normalize_features(
        graph,
        enhanced_features,
    )

    graph.x = normalized_features

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    graph = graph.to(DEVICE)

    # --------------------------------------------------------
    # MASKS
    # --------------------------------------------------------

    train_mask = graph.transaction_mask & graph.train_mask

    val_mask = graph.transaction_mask & graph.val_mask

    test_mask = graph.transaction_mask & graph.test_mask

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("\n==============================")
    print("DATASET")
    print("==============================")

    print(f"Training transactions: " f"{train_mask.sum().item():,}")

    print(f"Validation transactions: " f"{val_mask.sum().item():,}")

    print(f"Test transactions: " f"{test_mask.sum().item():,}")

    train_labels = graph.y[train_mask]

    fraud_count = (train_labels == 1).sum().item()

    normal_count = (train_labels == 0).sum().item()

    print(f"\nTraining normal: " f"{normal_count:,}")

    print(f"Training fraud: " f"{fraud_count:,}")

    print(f"Fraud rate: " f"{fraud_count / len(train_labels):.4%}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(graph.x.shape[1])

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    class_weights = torch.tensor(
        [
            1.0,
            FRAUD_CLASS_WEIGHT,
        ],
        dtype=torch.float,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(
        "\nFraud class weight:",
        FRAUD_CLASS_WEIGHT,
    )

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    best_pr_auc = -1.0

    best_epoch = 0

    patience_counter = 0

    best_state = None

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n==============================")
    print("TRAINING IMPROVED GRAPHSAGE")
    print("==============================")

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

        (
            val_roc,
            val_pr,
            _,
            _,
        ) = evaluate_split(
            model,
            graph,
            val_mask,
        )

        if val_pr > best_pr_auc:

            best_pr_auc = val_pr

            best_epoch = epoch

            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:

            patience_counter += 1

        if epoch % 5 == 0 or epoch == EPOCHS - 1:

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Val ROC {val_roc:.4f} | "
                f"Val PR {val_pr:.4f} | "
                f"Patience "
                f"{patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_state is None:

        raise RuntimeError("No valid model checkpoint " "was produced.")

    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        val_roc,
        val_pr,
        val_y,
        val_prob,
    ) = evaluate_split(
        model,
        graph,
        val_mask,
    )

    (
        validation_threshold,
        val_f1,
    ) = find_best_threshold(
        val_y,
        val_prob,
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": graph.x.shape[1],
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "fraud_class_weight": FRAUD_CLASS_WEIGHT,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_roc_auc": val_roc,
        "best_val_pr_auc": best_pr_auc,
        "validation_threshold": validation_threshold,
        "validation_f1": val_f1,
        "feature_names": feature_names,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "split_strategy": SPLIT_STRATEGY,
        "normalization_mean": normalization_mean.cpu(),
        "normalization_std": normalization_std.cpu(),
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print(f"\nSaved improved model: " f"{MODEL_PATH}")

    # --------------------------------------------------------
    # VALIDATION RESULTS
    # --------------------------------------------------------

    print("\n==============================")
    print("BEST VALIDATION RESULTS")
    print("==============================")

    print(f"Best epoch: {best_epoch}")

    print(f"Validation ROC-AUC: " f"{val_roc:.4f}")

    print(f"Validation PR-AUC: " f"{val_pr:.4f}")

    print(f"Validation threshold: " f"{validation_threshold:.4f}")

    print(f"Validation F1: " f"{val_f1:.4f}")

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    (
        test_roc,
        test_pr,
        test_y,
        test_prob,
    ) = evaluate_split(
        model,
        graph,
        test_mask,
    )

    test_predictions = (test_prob >= validation_threshold).astype(int)

    test_precision = precision_score(
        test_y,
        test_predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        test_y,
        test_predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        test_y,
        test_predictions,
        zero_division=0,
    )

    test_matrix = confusion_matrix(
        test_y,
        test_predictions,
    )

    print("\n==============================")
    print("IMPROVED GRAPHSAGE TEST RESULTS")
    print("==============================")

    print(f"ROC-AUC:   {test_roc:.4f}")

    print(f"PR-AUC:    {test_pr:.4f}")

    print(f"Precision: {test_precision:.4f}")

    print(f"Recall:    {test_recall:.4f}")

    print(f"F1:        {test_f1:.4f}")

    print(f"Threshold: " f"{validation_threshold:.4f}")

    print("\nConfusion Matrix:")

    print(test_matrix)

    # --------------------------------------------------------
    # SCORE DISTRIBUTION
    # --------------------------------------------------------

    print("\n==============================")
    print("TEST SCORE DISTRIBUTION")
    print("==============================")

    print(f"Min:    " f"{test_prob.min():.8f}")

    print(f"Max:    " f"{test_prob.max():.8f}")

    print(f"Mean:   " f"{test_prob.mean():.8f}")

    print(f"Median: " f"{np.median(test_prob):.8f}")

    print(f"P90:    " f"{np.percentile(test_prob, 90):.8f}")

    print(f"P99:    " f"{np.percentile(test_prob, 99):.8f}")

    print(f"P99.9:  " f"{np.percentile(test_prob, 99.9):.8f}")

    print("\n==============================")
    print("PHASE 10.6 COMPLETE")
    print("==============================")


if __name__ == "__main__":
    main()
