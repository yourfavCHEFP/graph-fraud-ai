import numpy as np

from src.training.metrics import SplitMetrics, calculate_split_metrics


def test_calculate_split_metrics_returns_named_metrics_and_inputs():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    result = calculate_split_metrics(y_true, y_prob)

    assert isinstance(result, SplitMetrics)
    assert result.roc_auc == 1.0
    assert result.pr_auc == 1.0
    assert result.y_true is y_true
    assert result.y_prob is y_prob
