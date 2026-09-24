"""Threshold selection utilities for fraud model evaluation."""

import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Return the threshold in [0.001, 0.999] with the best F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.linspace(0.001, 0.999, 999):
        predictions = (probabilities >= threshold).astype(int)
        current_f1 = f1_score(y_true, predictions, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = float(current_f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1
