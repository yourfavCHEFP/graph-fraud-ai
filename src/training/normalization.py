"""Feature normalization utilities for training workflows."""

import torch


def normalize_features(
    features: torch.Tensor,
    reference_mask: torch.Tensor | None = None,
    *,
    unbiased: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize features using statistics from an optional reference mask."""
    reference_features = features if reference_mask is None else features[reference_mask]
    if reference_features.numel() == 0:
        raise ValueError("No reference features found for normalization.")

    mean = reference_features.mean(dim=0, keepdim=True)
    std = reference_features.std(dim=0, keepdim=True, unbiased=unbiased)
    std = torch.where(std == 0, torch.ones_like(std), std)

    return (features - mean) / std, mean, std
