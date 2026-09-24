"""Class-weight calculation utilities for fraud training."""

import torch


def calculate_class_weights(
    labels: torch.Tensor,
    device: torch.device | str,
    *,
    max_fraud_weight: float = 5.0,
) -> torch.Tensor:
    """Return cross-entropy weights for binary normal/fraud labels."""
    fraud_count = (labels == 1).sum().float()
    normal_count = (labels == 0).sum().float()

    if fraud_count == 0:
        raise ValueError("Cannot calculate class weights without fraud labels.")

    fraud_weight = torch.sqrt(normal_count / fraud_count).clamp(max=max_fraud_weight)
    return torch.tensor(
        [1.0, fraud_weight.item()],
        dtype=torch.float,
        device=device,
    )
