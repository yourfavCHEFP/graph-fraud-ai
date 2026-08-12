"""
Prepare PyTorch Geometric fraud dataset.

Adds:
- Fraud labels
- Train/validation/test masks

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


def load_pyg_graph():

    print("\nLoading PyG graph...")

    graph = torch.load("data/graph/fraud_graph_pyg.pt", weights_only=False)

    print(graph)

    return graph


def load_networkx_graph():

    print("\nLoading NetworkX graph...")

    with open("data/graph/fraud_graph.pkl", "rb") as f:

        graph = pickle.load(f)

    print("NetworkX nodes:", graph.number_of_nodes())

    return graph


def load_labels():

    print("\nLoading fraud labels...")

    df = pd.read_parquet("data/processed/train_graph_features.parquet")

    print("Dataset:", df.shape)

    return df


def create_label_mapping(df):

    print("\nCreating label mapping...")

    labels = {}

    for _, row in df[["TransactionID", "isFraud"]].iterrows():

        node_id = f"transaction_{row.TransactionID}"

        labels[node_id] = int(row.isFraud)

    print("Labels created:", len(labels))

    return labels


def attach_labels(pyg_graph, nx_graph, label_mapping):

    print("\nAttaching labels...")

    nodes = list(nx_graph.nodes())

    y = torch.zeros(len(nodes), dtype=torch.long)

    fraud_count = 0

    for idx, node in enumerate(nodes):

        if node in label_mapping:

            y[idx] = label_mapping[node]

            fraud_count += label_mapping[node]

    pyg_graph.y = y

    print("Fraud nodes:", fraud_count)

    return pyg_graph


def create_masks(graph):

    print("\nCreating masks...")

    num_nodes = graph.num_nodes

    indices = list(range(num_nodes))

    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)

    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    graph.train_mask = torch.zeros(num_nodes, dtype=torch.bool)

    graph.val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    graph.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    graph.train_mask[train_idx] = True
    graph.val_mask[val_idx] = True
    graph.test_mask[test_idx] = True

    print("Train:", graph.train_mask.sum().item())

    print("Validation:", graph.val_mask.sum().item())

    print("Test:", graph.test_mask.sum().item())

    return graph


def main():

    print("==============================")
    print("PHASE 9 GNN DATASET PREPARATION")
    print("==============================")

    pyg_graph = load_pyg_graph()

    nx_graph = load_networkx_graph()

    df = load_labels()

    label_mapping = create_label_mapping(df)

    pyg_graph = attach_labels(pyg_graph, nx_graph, label_mapping)

    pyg_graph = create_masks(pyg_graph)

    torch.save(pyg_graph, "data/graph/fraud_graph_ready.pt")

    print("\nSaved:", "data/graph/fraud_graph_ready.pt")

    print("==============================")
    print("PHASE 9 COMPLETE")
    print("==============================")


if __name__ == "__main__":
    main()
