import numpy as np

from src.training.thresholds import find_best_threshold


def test_find_best_threshold_returns_high_f1_cutoff():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, score = find_best_threshold(y_true, probabilities)

    assert 0.2 < threshold <= 0.8
    assert score == 1.0


def test_find_best_threshold_is_deterministic_on_ties():
    y_true = np.array([0, 0])
    probabilities = np.array([0.5, 0.5])

    threshold, score = find_best_threshold(y_true, probabilities)

    assert threshold == 0.5
    assert score == 0.0
