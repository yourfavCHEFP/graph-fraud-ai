"""
Prepare PyTorch Geometric fraud dataset.

Adds:
- Fraud labels for transaction nodes
- Transaction-node mask
- Stratified train/validation/test masks

Entity nodes remain in the graph for message passing but are NOT
included as supervised prediction targets.

Input:
- data/graph/fraud_graph_pyg.pt
- data/processed/train_graph_features.parquet
- data/graph/fraud_graph.pkl

Output:
- data/graph/fraud_graph_ready.pt
"""

import pickle

import pandas as pd
import torch


PYG_GRAPH_PATH = "data/graph/fraud_graph_pyg.pt"
NETWORKX_GRAPH_PATH = "data/graph/fraud_graph.pkl"
TRANSACTION_DATA_PATH = (
    "data/processed/train_graph_features.parquet"
)
OUTPUT_PATH = "data/graph/fraud_graph_ready.pt"


def load_pyg_graph():

    print("\nLoading PyG graph...")

    graph = torch.load(
        PYG_GRAPH_PATH,
        weights_only=False
    )

    print(graph)

    return graph


def load_networkx_graph():

    print("\nLoading NetworkX graph...")

    with open(
        NETWORKX_GRAPH_PATH,
        "rb"
    ) as f:

        graph = pickle.load(f)

    print(
        "NetworkX nodes:",
        graph.number_of_nodes()
    )

    print(
        "NetworkX edges:",
        graph.number_of_edges()
    )

    return graph


def load_labels():

    print("\nLoading fraud labels...")

    df = pd.read_parquet(
        TRANSACTION_DATA_PATH
    )

    print(
        "Dataset:",
        df.shape
    )

    required_columns = [
        "TransactionID",
        "isFraud",
        "TransactionDT"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    return df


def create_label_mapping(df):

    print("\nCreating transaction label mapping...")

    labels = {}
    transaction_dt = {}

    for transaction_id, fraud_label, dt in zip(
        df["TransactionID"],
        df["isFraud"],
        df["TransactionDT"]
    ):

        node_id = (
            f"transaction_{int(transaction_id)}"
        )

        labels[node_id] = int(
            fraud_label
        )

        transaction_dt[node_id] = float(dt)

    print(
        "Transaction labels created:",
        len(labels)
    )

    fraud_count = sum(
        labels.values()
    )

    print(
        "Fraud transactions:",
        fraud_count
    )

    print(
        "Normal transactions:",
        len(labels) - fraud_count
    )

    return labels, transaction_dt


def attach_labels(
    pyg_graph,
    nx_graph,
    label_mapping,
    dt_mapping
):

    print("\nAttaching transaction labels...")

    nodes = list(
        nx_graph.nodes()
    )

    if len(nodes) != pyg_graph.num_nodes:

        raise ValueError(
            "NetworkX and PyG node counts do not match: "
            f"{len(nodes)} vs {pyg_graph.num_nodes}"
        )

    y = torch.zeros(
        pyg_graph.num_nodes,
        dtype=torch.long
    )

    transaction_dt = torch.zeros(
        pyg_graph.num_nodes,
        dtype=torch.float
    )

    transaction_mask = torch.zeros(
        pyg_graph.num_nodes,
        dtype=torch.bool
    )

    fraud_count = 0
    transaction_count = 0

    for idx, node in enumerate(nodes):

        if node in label_mapping:

            transaction_mask[idx] = True

            label = label_mapping[node]

            y[idx] = label

            transaction_dt[idx] = dt_mapping[node]

            transaction_count += 1
            fraud_count += label

    expected_transactions = len(
        label_mapping
    )

    if transaction_count != expected_transactions:

        raise ValueError(
            "Transaction-node alignment failed: "
            f"expected {expected_transactions}, "
            f"found {transaction_count}"
        )

    expected_fraud = sum(
        label_mapping.values()
    )

    if fraud_count != expected_fraud:

        raise ValueError(
            "Fraud-label count mismatch: "
            f"expected {expected_fraud}, "
            f"found {fraud_count}"
        )

    pyg_graph.y = y

    pyg_graph.transaction_mask = (
        transaction_mask
    )

    pyg_graph.transaction_dt = (
        transaction_dt
    )

    print(
        "Transaction nodes:",
        transaction_count
    )

    print(
        "Entity nodes:",
        pyg_graph.num_nodes - transaction_count
    )

    print(
        "Fraud transaction nodes:",
        fraud_count
    )

    print(
        "Normal transaction nodes:",
        transaction_count - fraud_count
    )

    return pyg_graph


def create_masks(graph):

    print(
        "\nCreating chronological transaction masks..."
    )

    # FIX (mentor review item 4): this used to be a RANDOM stratified
    # 70/15/15 split via sklearn's train_test_split(). IEEE-CIS is a
    # time-ordered fraud dataset (TransactionDT) -- a random split puts
    # future transactions in training and past transactions in test,
    # which is optimistic versus how the model will actually be used in
    # production (only the past is ever available to train on). The
    # split below sorts by TransactionDT and assigns the earliest period
    # to train, the next to validation, and the most recent to test.

    transaction_indices = (
        graph.transaction_mask
        .nonzero(as_tuple=False)
        .view(-1)
        .cpu()
        .numpy()
    )

    transaction_labels = (
        graph.y[
            graph.transaction_mask
        ]
        .cpu()
        .numpy()
    )

    transaction_timestamps = (
        graph.transaction_dt[
            graph.transaction_mask
        ]
        .cpu()
        .numpy()
    )

    if len(transaction_indices) == 0:

        raise ValueError(
            "No transaction nodes found."
        )

    if len(transaction_indices) != len(
        transaction_labels
    ):

        raise ValueError(
            "Transaction indices and labels "
            "have different lengths."
        )

    # Sort by TransactionDT ascending -- earliest transaction first.
    chronological_order = transaction_timestamps.argsort()

    sorted_node_indices = transaction_indices[chronological_order]
    sorted_labels = transaction_labels[chronological_order]
    sorted_timestamps = transaction_timestamps[chronological_order]

    n = len(sorted_node_indices)
    train_end = int(0.70 * n)
    val_end = train_end + int(0.15 * n)

    train_idx = sorted_node_indices[:train_end]
    val_idx = sorted_node_indices[train_end:val_end]
    test_idx = sorted_node_indices[val_end:]

    train_dt = sorted_timestamps[:train_end]
    val_dt = sorted_timestamps[train_end:val_end]
    test_dt = sorted_timestamps[val_end:]

    train_labels = sorted_labels[:train_end]
    val_labels = sorted_labels[train_end:val_end]
    test_labels = sorted_labels[val_end:]

    num_nodes = graph.num_nodes

    graph.train_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    graph.val_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    graph.test_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool
    )

    graph.train_mask[
        torch.tensor(
            train_idx,
            dtype=torch.long
        )
    ] = True

    graph.val_mask[
        torch.tensor(
            val_idx,
            dtype=torch.long
        )
    ] = True

    graph.test_mask[
        torch.tensor(
            test_idx,
            dtype=torch.long
        )
    ] = True

    # Safety checks:
    # No entity node should appear in a supervised mask.

    if (
        graph.train_mask
        & ~graph.transaction_mask
    ).any():

        raise ValueError(
            "Entity nodes detected in train mask."
        )

    if (
        graph.val_mask
        & ~graph.transaction_mask
    ).any():

        raise ValueError(
            "Entity nodes detected in validation mask."
        )

    if (
        graph.test_mask
        & ~graph.transaction_mask
    ).any():

        raise ValueError(
            "Entity nodes detected in test mask."
        )

    # Masks must not overlap.

    if (
        graph.train_mask
        & graph.val_mask
    ).any():

        raise ValueError(
            "Train and validation masks overlap."
        )

    if (
        graph.train_mask
        & graph.test_mask
    ).any():

        raise ValueError(
            "Train and test masks overlap."
        )

    if (
        graph.val_mask
        & graph.test_mask
    ).any():

        raise ValueError(
            "Validation and test masks overlap."
        )

    # Chronological ordering must hold: no split may contain a
    # transaction from AFTER any transaction in a later split.

    if len(train_dt) and len(val_dt):

        if train_dt.max() > val_dt.min():

            raise ValueError(
                "Chronological leakage detected: a training transaction "
                "occurs after a validation transaction."
            )

    if len(val_dt) and len(test_dt):

        if val_dt.max() > test_dt.min():

            raise ValueError(
                "Chronological leakage detected: a validation transaction "
                "occurs after a test transaction."
            )

    print(
        "Train:",
        graph.train_mask.sum().item()
    )

    print(
        "Validation:",
        graph.val_mask.sum().item()
    )

    print(
        "Test:",
        graph.test_mask.sum().item()
    )

    print(
        "Supervised nodes:",
        (
            graph.train_mask
            | graph.val_mask
            | graph.test_mask
        ).sum().item()
    )

    print(
        "Entity nodes excluded:",
        (
            ~graph.transaction_mask
        ).sum().item()
    )

    print("\nChronological split boundaries (TransactionDT):")

    print(
        f"  Train      : {train_dt.min():.0f} -> {train_dt.max():.0f}"
        if len(train_dt) else "  Train      : (empty)"
    )

    print(
        f"  Validation : {val_dt.min():.0f} -> {val_dt.max():.0f}"
        if len(val_dt) else "  Validation : (empty)"
    )

    print(
        f"  Test       : {test_dt.min():.0f} -> {test_dt.max():.0f}"
        if len(test_dt) else "  Test       : (empty)"
    )

    print("\nClass counts per split:")

    print(
        f"  Train      : fraud={train_labels.sum()}  legit={len(train_labels) - train_labels.sum()}"
    )

    print(
        f"  Validation : fraud={val_labels.sum()}  legit={len(val_labels) - val_labels.sum()}"
    )

    print(
        f"  Test       : fraud={test_labels.sum()}  legit={len(test_labels) - test_labels.sum()}"
    )

    return graph


def main():

    print("==============================")
    print("PHASE 9 GNN DATASET PREPARATION")
    print("==============================")

    pyg_graph = load_pyg_graph()

    nx_graph = load_networkx_graph()

    df = load_labels()

    label_mapping, dt_mapping = create_label_mapping(
        df
    )

    pyg_graph = attach_labels(
        pyg_graph,
        nx_graph,
        label_mapping,
        dt_mapping
    )

    pyg_graph = create_masks(
        pyg_graph
    )

    torch.save(
        pyg_graph,
        OUTPUT_PATH
    )

    print(
        "\nSaved:",
        OUTPUT_PATH
    )

    print("==============================")
    print("PHASE 9 COMPLETE")
    print("==============================")


if __name__ == "__main__":

    main()
