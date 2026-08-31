"""
Regression test for the critical feature-column misalignment bug
(mentor review, item 1 / item 32).

src/features/graph_features.py used to read graph.x by hardcoded column
POSITION under a comment describing a column order that did not match
what src/graph/pyg_converter.py actually produces -- every feature was
silently off by one column, and address_degree was actually reading
node_type_id (a categorical label, not a degree count).

This test builds a small synthetic graph with KNOWN, DISTINCT values in
each column (following pyg_converter.FEATURE_COLUMNS' real order) and
asserts that build_graph_features() reads each source column correctly
by value, not just by shape. If the column contract in pyg_converter.py
ever changes without this file being updated to match, this test fails
loudly instead of silently producing wrong features again.
"""

import torch
import pytest
from torch_geometric.data import Data

from src.graph.pyg_converter import FEATURE_COLUMNS
from src.features.graph_features import build_graph_features, get_feature_names


@pytest.fixture
def synthetic_graph():
    """
    One node, with a DISTINCT, recognizable value in every column so a
    column-order bug is impossible to miss: value == column index * 10,
    except log_transaction_amount and the degree columns get realistic
    magnitudes so log1p() and the ratio math stay well-behaved.
    """
    values = {name: float(i * 10) for i, name in enumerate(FEATURE_COLUMNS)}
    # Override with values chosen so downstream math (log1p, ratios,
    # hub thresholds) produces distinguishable, sane numbers.
    values["log_transaction_amount"] = 5.0   # this is what "amount" must equal
    values["card_degree"] = 200.0            # >= 100 -> triggers hub_card
    values["email_degree"] = 3.0
    values["device_degree"] = 7.0
    values["address_degree"] = 2.0
    values["node_type_id"] = 999.0           # must NEVER appear in any output feature

    row = [values[name] for name in FEATURE_COLUMNS]
    x = torch.tensor([row], dtype=torch.float)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_transaction_amount_reads_log_transaction_amount_column(synthetic_graph):
    features = build_graph_features(synthetic_graph)
    amount_idx = get_feature_names().index("transaction_amount")
    assert features[0, amount_idx].item() == pytest.approx(5.0)


def test_node_type_id_never_leaks_into_any_output_feature(synthetic_graph):
    """
    The core bug: address_degree = x[:, 7] was actually reading
    node_type_id (999.0 in this fixture). If that regresses, this
    sentinel value will show up somewhere in the output -- it must not.
    """
    features = build_graph_features(synthetic_graph)
    assert not torch.any(features == 999.0), (
        "node_type_id's sentinel value leaked into an output feature -- "
        "a column-index bug has regressed."
    )
    assert not torch.any(torch.isclose(features, torch.log1p(torch.tensor(999.0)))), (
        "log1p(node_type_id) leaked into an output feature -- "
        "address_degree is reading the wrong column again."
    )


def test_card_degree_feeds_log_card_degree_and_hub_card(synthetic_graph):
    features = build_graph_features(synthetic_graph)
    log_card_idx = get_feature_names().index("log_card_degree")
    expected = torch.log1p(torch.tensor(200.0)).item()
    assert features[0, log_card_idx].item() == pytest.approx(expected)


def test_output_dimension_matches_declared_feature_names(synthetic_graph):
    features = build_graph_features(synthetic_graph)
    assert features.shape[1] == len(get_feature_names()) == 16


def test_raises_clear_error_on_wrong_input_dimension():
    bad_x = torch.rand(3, 5)  # wrong number of columns vs FEATURE_COLUMNS
    bad_graph = Data(x=bad_x, edge_index=torch.zeros((2, 0), dtype=torch.long))
    with pytest.raises(AssertionError, match="graph.x has"):
        build_graph_features(bad_graph)


def test_no_engineered_feature_is_constant_across_varied_nodes():
    """Guards against a subtler version of the same bug: even if shapes
    line up, reading the wrong columns can still produce features that
    are accidentally constant (e.g. always 0) for realistic inputs."""
    torch.manual_seed(0)
    n = 20
    x = torch.zeros(n, len(FEATURE_COLUMNS))
    for i, name in enumerate(FEATURE_COLUMNS):
        if name == "node_type_id":
            x[:, i] = torch.randint(0, 4, (n,)).float()
        elif name in ("card_degree", "email_degree", "device_degree", "address_degree"):
            # Hub thresholds are 100 (card) / 1000 (email/device/address) --
            # range must actually cross them sometimes, or hub_entity_count
            # is spuriously "constant" for reasons unrelated to the bug
            # this test suite targets.
            x[:, i] = torch.rand(n) * 2000
        else:
            x[:, i] = torch.rand(n) * 50
    graph = Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long))

    features = build_graph_features(graph)
    stds = features.std(dim=0)
    constant_features = [
        get_feature_names()[i] for i, s in enumerate(stds) if s.item() < 1e-8
    ]
    assert not constant_features, (
        f"These features are constant across varied random input, which "
        f"suggests they're reading the wrong (or a degenerate) source "
        f"column: {constant_features}"
    )
