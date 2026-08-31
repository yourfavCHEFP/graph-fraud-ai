"""
Phase 10.9 - Graph Fraud Feature Engineering.

Build transaction-level graph features.

CRITICAL FIX (mentor review, item 1): this file previously read graph.x
by hardcoded column position under a comment claiming:

    0 node_type_id
    1 degree
    2 transaction_entity_count
    3 log_transaction_amount
    4 card_degree
    5 email_degree
    6 device_degree
    7 address_degree

But src/graph/pyg_converter.py -- the file that actually produces
graph.x -- places the columns in a DIFFERENT order:

    0 degree
    1 transaction_entity_count
    2 log_transaction_amount
    3 card_degree
    4 email_degree
    5 device_degree
    6 address_degree
    7 node_type_id

Every column read here was off by one relative to the real layout. Most
seriously, `address_degree = x[:, 7]` was actually reading node_type_id
(a categorical 0/1/2/3 label), not a real degree count -- meaning
`log_address_degree`, `entity_degree_min` (via the entity_stack min),
and `address_ratio` were all computed from a node-type label, and
`transaction_amount` (`amount = x[:, 3]`) was actually reading
card_degree instead of log_transaction_amount.

FIX: column positions are now looked up BY NAME from
src.graph.pyg_converter.FEATURE_COLUMNS -- the single authoritative
schema -- instead of being hardcoded twice in two files that can (and
did) silently drift apart. An assertion at import time fails loudly if
that schema ever changes shape without this file being updated to match.

Features (unchanged output contract -- same 16 names/order as before):
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

from src.graph.pyg_converter import FEATURE_COLUMNS

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

# Source columns this file actually reads from graph.x, bound by NAME to
# pyg_converter.FEATURE_COLUMNS rather than by unexplained numeric
# position. If pyg_converter.py ever reorders/renames its columns without
# this file being updated, _SOURCE_INDEX below raises a clear ValueError
# at import time (via .index()) instead of silently reading the wrong
# column again.
_REQUIRED_SOURCE_COLUMNS = [
    "log_transaction_amount",
    "card_degree",
    "email_degree",
    "device_degree",
    "address_degree",
]

_missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in FEATURE_COLUMNS]
if _missing:
    raise AssertionError(
        f"src.graph.pyg_converter.FEATURE_COLUMNS is missing expected "
        f"column(s) {_missing} -- this file's feature extraction depends "
        f"on them by name. Update _REQUIRED_SOURCE_COLUMNS here to match "
        f"any intentional schema change in pyg_converter.py."
    )

_SOURCE_INDEX = {name: FEATURE_COLUMNS.index(name) for name in _REQUIRED_SOURCE_COLUMNS}


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

    if x.shape[1] != len(FEATURE_COLUMNS):
        raise AssertionError(
            f"graph.x has {x.shape[1]} columns but the declared schema "
            f"(src.graph.pyg_converter.FEATURE_COLUMNS) expects "
            f"{len(FEATURE_COLUMNS)}: {FEATURE_COLUMNS}. This graph was "
            f"likely built with a different/older feature-column layout -- "
            f"rebuild it with the current pyg_converter.py before running "
            f"inference or training against it."
        )

    # FIX: read by name via _SOURCE_INDEX, not by hardcoded position.
    amount = x[:, _SOURCE_INDEX["log_transaction_amount"]]

    card_degree = x[:, _SOURCE_INDEX["card_degree"]]

    email_degree = x[:, _SOURCE_INDEX["email_degree"]]

    device_degree = x[:, _SOURCE_INDEX["device_degree"]]

    address_degree = x[:, _SOURCE_INDEX["address_degree"]]

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
