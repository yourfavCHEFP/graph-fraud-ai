"""
Phase 10.9 - Graph Fraud Feature Engineering.

Build transaction-level graph features from the original 8 graph features.

Features:
    0  log_transaction_amount
    1  log_card_degree
    2  log_email_degree
    3  log_device_degree
    4  log_address_degree
    5  entity_degree_mean
    6  entity_degree_max
    7  entity_degree_min
    8  entity_degree_std
    9  card_ratio
    10 email_ratio
    11 device_ratio
    12 address_ratio
    13 entity_concentration
    14 degree_imbalance
    15 hub_entity_count

Design requirements:
    - No fraud labels are used.
    - No target-derived statistics are used.
    - No global dataset statistics are used here.
    - The feature builder must remain deterministic.
    - The feature names must match the tensor column order exactly.

Input graph.x layout:
    0 node_type_id
    1 degree
    2 transaction_entity_count
    3 log_transaction_amount
    4 card_degree
    5 email_degree
    6 device_degree
    7 address_degree
"""

import torch

ORIGINAL_FEATURE_DIM = 8

FEATURE_NAMES = [
    "log_transaction_amount",
    "log_card_degree",
    "log_email_degree",
    "log_device_degree",
    "log_address_degree",
    "entity_degree_mean",
    "entity_degree_max",
    "entity_degree_min",
    "entity_degree_std",
    "card_ratio",
    "email_ratio",
    "device_ratio",
    "address_ratio",
    "entity_concentration",
    "degree_imbalance",
    "hub_entity_count",
]


ENHANCED_FEATURE_DIM = len(FEATURE_NAMES)


# ============================================================
# FEATURE NAME ACCESS
# ============================================================


def get_feature_names():
    """Return the feature names in exact tensor column order."""

    return FEATURE_NAMES.copy()


# ============================================================
# FEATURE BUILDING
# ============================================================


def build_graph_features(graph):
    """
    Build leakage-safe transaction-level graph features.

    The function expects graph.x to contain exactly the original 8
    graph features produced by the graph preparation pipeline.

    No fraud labels, train/validation/test labels, or global dataset
    statistics are used to construct these features.
    """

    print("\n==============================")
    print("BUILDING GRAPH FEATURES")
    print("==============================")

    if not hasattr(graph, "x"):
        raise ValueError("Graph does not contain an x feature tensor.")

    x = graph.x.float()

    if x.ndim != 2:
        raise ValueError(
            "Expected graph.x to be a 2D tensor, " f"got shape {tuple(x.shape)}."
        )

    print("Original feature dimension:", x.shape[1])

    if x.shape[1] != ORIGINAL_FEATURE_DIM:
        raise ValueError(
            "Unexpected original graph feature dimension. "
            f"Expected {ORIGINAL_FEATURE_DIM}, got {x.shape[1]}."
        )

    if not torch.isfinite(x).all():
        raise ValueError("Original graph features contain NaN or infinite values.")

    # Original graph layout
    #
    # 0 node_type_id
    # 1 degree
    # 2 transaction_entity_count
    # 3 log_transaction_amount
    # 4 card_degree
    # 5 email_degree
    # 6 device_degree
    # 7 address_degree

    amount = x[:, 3]
    card_degree = x[:, 4]
    email_degree = x[:, 5]
    device_degree = x[:, 6]
    address_degree = x[:, 7]

    # Entity degrees should be non-negative counts. Guard against a
    # malformed graph before applying log1p and ratio operations.
    entity_degrees = torch.stack(
        [
            card_degree,
            email_degree,
            device_degree,
            address_degree,
        ],
        dim=1,
    )

    if (entity_degrees < 0).any():
        raise ValueError("Entity degree features cannot contain negative values.")

    # ------------------------------------------------------------
    # LOG ENTITY DEGREE
    # ------------------------------------------------------------

    log_entity_degrees = torch.log1p(entity_degrees)

    log_card = log_entity_degrees[:, 0]
    log_email = log_entity_degrees[:, 1]
    log_device = log_entity_degrees[:, 2]
    log_address = log_entity_degrees[:, 3]

    # ------------------------------------------------------------
    # ENTITY STATISTICS
    # ------------------------------------------------------------

    degree_mean = log_entity_degrees.mean(dim=1)

    degree_max = log_entity_degrees.max(dim=1).values

    degree_min = log_entity_degrees.min(dim=1).values

    # These four entity types are the complete population of entity
    # degree features for a transaction, so use population std.
    degree_std = log_entity_degrees.std(
        dim=1,
        unbiased=False,
    )

    # ------------------------------------------------------------
    # ENTITY RATIOS
    # ------------------------------------------------------------

    total_degree = log_entity_degrees.sum(dim=1)

    safe_total_degree = total_degree.clamp_min(1e-6)

    card_ratio = log_card / safe_total_degree
    email_ratio = log_email / safe_total_degree
    device_ratio = log_device / safe_total_degree
    address_ratio = log_address / safe_total_degree

    # ------------------------------------------------------------
    # STRUCTURAL CONCENTRATION
    # ------------------------------------------------------------

    concentration = degree_max / safe_total_degree

    imbalance = degree_std / degree_mean.clamp_min(1e-6)

    # ------------------------------------------------------------
    # HUB FEATURES
    # ------------------------------------------------------------

    hub_card = (card_degree >= 100).float()
    hub_email = (email_degree >= 1000).float()
    hub_device = (device_degree >= 1000).float()
    hub_address = (address_degree >= 1000).float()

    hub_count = hub_card + hub_email + hub_device + hub_address

    # ------------------------------------------------------------
    # FINAL FEATURES
    # ------------------------------------------------------------

    features = torch.stack(
        [
            amount,
            log_card,
            log_email,
            log_device,
            log_address,
            degree_mean,
            degree_max,
            degree_min,
            degree_std,
            card_ratio,
            email_ratio,
            device_ratio,
            address_ratio,
            concentration,
            imbalance,
            hub_count,
        ],
        dim=1,
    )

    if features.shape[1] != ENHANCED_FEATURE_DIM:
        raise RuntimeError(
            "Enhanced feature dimension mismatch. "
            f"Expected {ENHANCED_FEATURE_DIM}, got {features.shape[1]}."
        )

    if not torch.isfinite(features).all():
        raise ValueError("Enhanced graph features contain NaN or infinite values.")

    print("Enhanced feature dimension:", features.shape[1])
    print("Feature tensor:", features.shape)

    return features
