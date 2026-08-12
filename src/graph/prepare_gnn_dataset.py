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
from sklearn.model_selection import train_test_split


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
        "isFraud"
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

    for transaction_id, fraud_label in zip(
        df["TransactionID"],
        df["isFraud"]
    ):

        node_id = (
            f"transaction_{int(transaction_id)}"
        )

        labels[node_id] = int(
            fraud_label
        )

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

    return labels


def attach_labels(
    pyg_graph,
    nx_graph,
    label_mapping
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
        "\nCreating stratified transaction masks..."
    )

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

    # 70% train, 15% validation, 15% test
    train_idx, temp_idx = train_test_split(
        transaction_indices,
        test_size=0.30,
        random_state=42,
        stratify=transaction_labels
    )

    temp_labels = (
        graph.y[
            torch.tensor(
                temp_idx,
                dtype=torch.long
            )
        ]
        .cpu()
        .numpy()
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels
    )

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

    return graph


def main():

    print("==============================")
    print("PHASE 9 GNN DATASET PREPARATION")
    print("==============================")

    pyg_graph = load_pyg_graph()

    nx_graph = load_networkx_graph()

    df = load_labels()

    label_mapping = create_label_mapping(
        df
    )

    pyg_graph = attach_labels(
        pyg_graph,
        nx_graph,
        label_mapping
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
