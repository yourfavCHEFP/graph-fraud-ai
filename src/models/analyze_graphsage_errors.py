"""
Phase 10.8 - GraphSAGE Error Analysis.

Uses the exact feature mapping and training-only normalization statistics
saved inside the GraphSAGE training checkpoint.

The analysis deliberately uses the same feature builder as training so
feature order cannot silently drift between training and error analysis.
"""

import os

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.features.graph_features import (
    build_graph_features,
    get_feature_names,
)
from src.models.gnn_model import (
    FraudGraphSAGE,
)

GRAPH_PATH = "data/graph/fraud_graph_ready.pt"
MODEL_PATH = "models/graphsage_improved.pt"
REPORT_PATH = "reports/graphsage_error_analysis.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    return graph


# ============================================================
# LOAD CHECKPOINT
# ============================================================


def load_checkpoint():

    print("\n==============================")
    print("LOADING GRAPHSAGE MODEL")
    print("==============================")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    required_keys = [
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "dropout",
        "best_epoch",
        "best_val_pr_auc",
        "validation_threshold",
        "feature_names",
        "normalization_mean",
        "normalization_std",
    ]

    missing = [key for key in required_keys if key not in checkpoint]

    if missing:
        raise ValueError(
            "GraphSAGE checkpoint is missing required fields: "
            + ", ".join(missing)
            + ". Retrain GraphSAGE using the corrected training script."
        )

    model = FraudGraphSAGE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=2,
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    feature_names = list(checkpoint["feature_names"])
    normalization_mean = checkpoint["normalization_mean"].float()
    normalization_std = checkpoint["normalization_std"].float()

    if normalization_mean.ndim != 2:
        raise ValueError(
            "Invalid normalization_mean shape: " f"{normalization_mean.shape}"
        )

    if normalization_std.ndim != 2:
        raise ValueError(
            "Invalid normalization_std shape: " f"{normalization_std.shape}"
        )

    input_dim = int(checkpoint["input_dim"])

    if normalization_mean.shape[1] != input_dim:
        raise ValueError(
            "Normalization mean dimension does not match model input dimension."
        )

    if normalization_std.shape[1] != input_dim:
        raise ValueError(
            "Normalization std dimension does not match model input dimension."
        )

    if torch.any(normalization_std <= 0):
        raise ValueError("Checkpoint normalization_std contains non-positive values.")

    generated_feature_names = get_feature_names()

    if feature_names != generated_feature_names:
        raise ValueError(
            "Checkpoint feature mapping does not match the current feature builder.\n"
            f"Checkpoint: {feature_names}\n"
            f"Current:    {generated_feature_names}\n"
            "Retrain GraphSAGE after correcting the feature mapping."
        )

    if len(feature_names) != input_dim:
        raise ValueError(
            "Checkpoint feature-name count does not match model input dimension: "
            f"{len(feature_names)} vs {input_dim}."
        )

    print("Model loaded successfully.")
    print("Input dimension:", input_dim)
    print("Hidden dimension:", checkpoint["hidden_dim"])
    print("Dropout:", checkpoint["dropout"])
    print("Best epoch:", checkpoint["best_epoch"])
    print("Best validation PR-AUC:", checkpoint["best_val_pr_auc"])
    print("Validation threshold:", checkpoint["validation_threshold"])
    print("Checkpoint feature count:", len(feature_names))

    return (
        model,
        float(checkpoint["validation_threshold"]),
        feature_names,
        normalization_mean,
        normalization_std,
    )


# ============================================================
# APPLY SAVED NORMALIZATION
# ============================================================


def normalize_with_checkpoint(
    features,
    normalization_mean,
    normalization_std,
):

    print("\n==============================")
    print("APPLYING CHECKPOINT NORMALIZATION")
    print("==============================")

    if features.shape[1] != normalization_mean.shape[1]:
        raise ValueError(
            "Feature dimension mismatch.\n"
            f"Generated features: {features.shape[1]}\n"
            f"Checkpoint expects: {normalization_mean.shape[1]}"
        )

    normalized = (features - normalization_mean) / normalization_std

    if not torch.isfinite(normalized).all():
        raise ValueError("Normalized features contain NaN or infinite values.")

    print("Exact training normalization statistics loaded from checkpoint.")

    return normalized


# ============================================================
# GENERATE PREDICTIONS
# ============================================================


def generate_predictions(
    model,
    graph,
    threshold,
):

    print("\n==============================")
    print("GENERATING TEST PREDICTIONS")
    print("==============================")

    with torch.no_grad():
        output = model(
            graph.x,
            graph.edge_index,
        )

        probability = torch.softmax(
            output,
            dim=1,
        )[:, 1]

    test_mask = graph.transaction_mask & graph.test_mask

    y_true = graph.y[test_mask].detach().cpu().numpy()
    y_prob = probability[test_mask].detach().cpu().numpy()
    predictions = (y_prob >= threshold).astype(int)

    return (
        y_true,
        y_prob,
        predictions,
        test_mask,
    )


# ============================================================
# ERROR SUMMARY
# ============================================================


def calculate_errors(
    y_true,
    predictions,
):

    print("\n==============================")
    print("ERROR SUMMARY")
    print("==============================")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    print(f"True negatives:  {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives:  {tp:,}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1:              {f1:.4f}")

    return tn, fp, fn, tp


# ============================================================
# SCORE STATISTICS
# ============================================================


def score_statistics(
    name,
    scores,
):

    if len(scores) == 0:
        return

    print(f"\n{name}")
    print(f"Count:   {len(scores):,}")
    print(f"Mean:    {scores.mean():.8f}")
    print(f"Median:  {np.median(scores):.8f}")
    print(f"P90:     {np.percentile(scores, 90):.8f}")
    print(f"P99:     {np.percentile(scores, 99):.8f}")
    print(f"Max:     {scores.max():.8f}")


# ============================================================
# ERROR SCORE ANALYSIS
# ============================================================


def analyze_error_scores(
    y_true,
    y_prob,
    predictions,
):

    print("\n==============================")
    print("ERROR SCORE ANALYSIS")
    print("==============================")

    false_positive = (predictions == 1) & (y_true == 0)
    false_negative = (predictions == 0) & (y_true == 1)
    true_positive = (predictions == 1) & (y_true == 1)
    true_negative = (predictions == 0) & (y_true == 0)

    score_statistics(
        "FALSE POSITIVE SCORES",
        y_prob[false_positive],
    )

    score_statistics(
        "FALSE NEGATIVE SCORES",
        y_prob[false_negative],
    )

    score_statistics(
        "TRUE POSITIVE SCORES",
        y_prob[true_positive],
    )

    score_statistics(
        "TRUE NEGATIVE SCORES",
        y_prob[true_negative],
    )


# ============================================================
# FEATURE ERROR ANALYSIS
# ============================================================


def analyze_feature_groups(
    graph,
    test_mask,
    y_true,
    predictions,
    feature_names,
):

    print("\n==============================")
    print("FEATURE ERROR ANALYSIS")
    print("==============================")

    x = graph.x.detach().cpu().numpy()

    test_indices = test_mask.detach().cpu().numpy()
    test_features = x[test_indices]

    if test_features.shape[1] != len(feature_names):
        raise ValueError(
            "Feature analysis dimension mismatch.\n"
            f"Test feature matrix contains {test_features.shape[1]} columns.\n"
            f"Feature names contain {len(feature_names)} names."
        )

    false_positive = (predictions == 1) & (y_true == 0)
    false_negative = (predictions == 0) & (y_true == 1)
    true_positive = (predictions == 1) & (y_true == 1)

    groups = {
        "FALSE POSITIVE": false_positive,
        "FALSE NEGATIVE": false_negative,
        "TRUE POSITIVE": true_positive,
    }

    for group_name, mask in groups.items():

        print(f"\n{group_name}")

        if mask.sum() == 0:
            print("No observations.")
            continue

        group_features = test_features[mask]

        for index, name in enumerate(feature_names):

            values = group_features[:, index]

            print(
                f"{name:28s} "
                f"mean={values.mean():.4f} "
                f"median={np.median(values):.4f}"
            )


# ============================================================
# REPORT
# ============================================================


def write_report(
    tn,
    fp,
    fn,
    tp,
    threshold,
    feature_names,
):

    precision = precision_score(
        [0] * tn + [0] * fp + [1] * fn + [1] * tp,
        [0] * tn + [1] * fp + [0] * fn + [1] * tp,
        zero_division=0,
    )

    recall = recall_score(
        [0] * tn + [0] * fp + [1] * fn + [1] * tp,
        [0] * tn + [1] * fp + [0] * fn + [1] * tp,
        zero_division=0,
    )

    f1 = f1_score(
        [0] * tn + [0] * fp + [1] * fn + [1] * tp,
        [0] * tn + [1] * fp + [0] * fn + [1] * tp,
        zero_division=0,
    )

    lines = [
        "PHASE 10.8 GRAPHSAGE ERROR ANALYSIS\n",
        "Validation Threshold: " f"{threshold:.4f}\n",
        f"True Negatives: {tn:,}\n",
        f"False Positives: {fp:,}\n",
        f"False Negatives: {fn:,}\n",
        f"True Positives: {tp:,}\n",
        f"Precision: {precision:.4f}\n",
        f"Recall: {recall:.4f}\n",
        f"F1: {f1:.4f}\n",
        f"Feature Count: {len(feature_names)}\n",
        "\nFeature Names:\n",
    ]

    lines.extend(f"{index}: {name}\n" for index, name in enumerate(feature_names))

    lines.append("\nError analysis completed successfully.\n")

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.writelines(lines)

    print(f"\nSaved report: {REPORT_PATH}")


# ============================================================
# MAIN
# ============================================================


def main():

    print("==============================")
    print("PHASE 10.8 GRAPHSAGE ERROR ANALYSIS")
    print("==============================")

    print(f"\nDevice: {DEVICE}")

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    graph = load_graph()

    (
        model,
        threshold,
        feature_names,
        normalization_mean,
        normalization_std,
    ) = load_checkpoint()

    # --------------------------------------------------------
    # BUILD SAME FEATURES AS TRAINING
    # --------------------------------------------------------

    enhanced_features = build_graph_features(graph)

    generated_feature_names = get_feature_names()

    print(
        "\nGenerated feature dimension:",
        enhanced_features.shape[1],
    )

    print(
        "Checkpoint feature dimension:",
        len(feature_names),
    )

    if enhanced_features.shape[1] != len(feature_names):
        raise ValueError(
            "The feature builder and GraphSAGE checkpoint do not use "
            "the same number of features."
        )

    if generated_feature_names != feature_names:
        raise ValueError(
            "The feature builder and GraphSAGE checkpoint use different "
            "feature mappings. Retrain GraphSAGE after correcting the mapping."
        )

    # --------------------------------------------------------
    # NORMALIZE USING TRAINING CHECKPOINT
    # --------------------------------------------------------

    normalized_features = normalize_with_checkpoint(
        enhanced_features,
        normalization_mean,
        normalization_std,
    )

    graph.x = normalized_features
    graph = graph.to(DEVICE)

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    (
        y_true,
        y_prob,
        predictions,
        test_mask,
    ) = generate_predictions(
        model,
        graph,
        threshold,
    )

    # --------------------------------------------------------
    # ERROR SUMMARY
    # --------------------------------------------------------

    tn, fp, fn, tp = calculate_errors(
        y_true,
        predictions,
    )

    # --------------------------------------------------------
    # SCORE ANALYSIS
    # --------------------------------------------------------

    analyze_error_scores(
        y_true,
        y_prob,
        predictions,
    )

    # --------------------------------------------------------
    # FEATURE ANALYSIS
    # --------------------------------------------------------

    analyze_feature_groups(
        graph,
        test_mask,
        y_true,
        predictions,
        feature_names,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    write_report(
        tn,
        fp,
        fn,
        tp,
        threshold,
        feature_names,
    )

    print("\n==============================")
    print("PHASE 10.8 COMPLETE")
    print("==============================")


if __name__ == "__main__":
    main()
