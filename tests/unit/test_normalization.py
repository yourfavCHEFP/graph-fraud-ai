import pytest
import torch

from src.training.normalization import normalize_features


def test_normalize_features_uses_only_reference_rows():
    features = torch.tensor([[1.0, 10.0], [3.0, 14.0], [100.0, 200.0]])
    mask = torch.tensor([True, True, False])

    normalized, mean, std = normalize_features(features, mask, unbiased=False)

    assert mean.tolist() == [[2.0, 12.0]]
    assert std.tolist() == [[1.0, 2.0]]
    assert normalized[0].tolist() == pytest.approx([-1.0, -1.0])
    assert normalized[2].tolist() == pytest.approx([98.0, 94.0])


def test_normalize_features_rejects_empty_reference_mask():
    features = torch.ones((2, 2))

    with pytest.raises(ValueError, match="No reference features"):
        normalize_features(features, torch.tensor([False, False]))
