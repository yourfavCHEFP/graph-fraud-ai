"""
Phase 10.9 - Graph Fraud Feature Engineering.

Build transaction-level graph features.

Features:
    0 transaction_amount
    1 log_card_degree
    2 log_email_degree
    3 log_device_degree
    4 log_address_degree
    5 entity_degree_mean
    6 entity_degree_max
    7 entity_degree_min
    8 entity_degree_std
    9 card_ratio
    10 email_ratio
    11 device_ratio
    12 address_ratio
    13 entity_concentration
    14 degree_imbalance
    15 hub_entity_count

No fraud labels are used.
No global dataset statistics are used.
"""

import torch

FEATURE_NAMES = [
    "transaction_amount",
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


def get_feature_names():

    return FEATURE_NAMES.copy()


def build_graph_features(graph):

    print("\n==============================")
    print("BUILDING GRAPH FEATURES")
    print("==============================")

    x = graph.x.float()

    print(
        "Original feature dimension:",
        x.shape[1],
    )

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

    # -----------------------------
    # LOG DEGREE FEATURES
    # -----------------------------

    log_card = torch.log1p(card_degree)

    log_email = torch.log1p(email_degree)

    log_device = torch.log1p(device_degree)

    log_address = torch.log1p(address_degree)

    entity_stack = torch.stack(
        [
            log_card,
            log_email,
            log_device,
            log_address,
        ],
        dim=1,
    )

    # -----------------------------
    # ENTITY STATISTICS
    # -----------------------------

    degree_mean = entity_stack.mean(dim=1)

    degree_max = entity_stack.max(dim=1).values

    degree_min = entity_stack.min(dim=1).values

    degree_std = entity_stack.std(dim=1)

    # -----------------------------
    # ENTITY RATIOS
    # -----------------------------

    total_degree = log_card + log_email + log_device + log_address + 1e-6

    card_ratio = log_card / total_degree

    email_ratio = log_email / total_degree

    device_ratio = log_device / total_degree

    address_ratio = log_address / total_degree

    # -----------------------------
    # STRUCTURE FEATURES
    # -----------------------------

    concentration = degree_max / total_degree

    imbalance = degree_std / (degree_mean + 1e-6)

    # -----------------------------
    # HUB FEATURES
    # -----------------------------

    hub_card = (card_degree >= 100).float()

    hub_email = (email_degree >= 1000).float()

    hub_device = (device_degree >= 1000).float()

    hub_address = (address_degree >= 1000).float()

    hub_count = hub_card + hub_email + hub_device + hub_address

    # -----------------------------
    # FINAL FEATURES
    # -----------------------------

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

    print(
        "Enhanced feature dimension:",
        features.shape[1],
    )

    print(
        "Feature tensor:",
        features.shape,
    )

    return features
