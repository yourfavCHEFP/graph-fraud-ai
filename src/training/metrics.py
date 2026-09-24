"""Reusable evaluation metric contracts for training workflows."""

from typing import NamedTuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class SplitMetrics(NamedTuple):
    """Ranking metrics and the values used to calculate them."""

    roc_auc: float
    pr_auc: float
    y_true: np.ndarray
    y_prob: np.ndarray


def calculate_split_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> SplitMetrics:
    """Calculate ranking metrics for one evaluated graph split."""
    return SplitMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        y_true=y_true,
        y_prob=y_prob,
    )
