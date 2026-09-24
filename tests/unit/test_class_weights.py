from types import SimpleNamespace

import pytest
import torch

from src.models import optimize_gcn, train_gnn
from src.training.class_weights import calculate_class_weights


def test_calculate_class_weights_applies_sqrt_and_cap():
    labels = torch.tensor([0, 0, 0, 0, 1])

    weights = calculate_class_weights(labels, "cpu")

    assert weights.tolist() == pytest.approx([1.0, 2.0])


def test_calculate_class_weights_rejects_no_fraud_labels():
    with pytest.raises(ValueError, match="without fraud labels"):
        calculate_class_weights(torch.tensor([0, 0]), "cpu")


def test_legacy_model_scripts_reject_no_fraud_labels():
    graph = SimpleNamespace(
        y=torch.tensor([0, 0, 0, 0]),
        train_mask=torch.tensor([True, True, True, True]),
        transaction_mask=torch.tensor([True, True, True, True]),
    )

    with pytest.raises(ValueError, match="without fraud labels"):
        train_gnn.calculate_class_weights(graph, "cpu")

    with pytest.raises(ValueError, match="without fraud labels"):
        optimize_gcn.calculate_class_weights(graph, "cpu")
