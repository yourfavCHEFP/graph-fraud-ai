"""
Convert fraud graph into PyTorch Geometric format.

Input:
- data/graph/fraud_graph.pkl
- data/graph/node_features.parquet

Output:
- data/graph/fraud_graph_pyg.pt
"""

import pickle

import pandas as pd
import torch
from torch_geometric.data import Data


def load_graph():

    print("\nLoading NetworkX graph...")

    with open("data/graph/fraud_graph.pkl", "rb") as f:

        graph = pickle.load(f)

    print("Nodes:", graph.number_of_nodes())

    print("Edges:", graph.number_of_edges())

    return graph


def load_features():

    print("\nLoading graph features...")

    df = pd.read_parquet("data/graph/node_features.parquet")

    print("Feature shape:", df.shape)

    return df


def build_pyg_graph(graph, feature_df):

    print("\nBuilding PyTorch Geometric graph...")

    nodes = list(graph.nodes())

    node_mapping = {node: idx for idx, node in enumerate(nodes)}

    print("Total mapped nodes:", len(node_mapping))

    feature_columns = [
        "degree",
        "transaction_entity_count",
        "fraud_neighbor_count",
        "fraud_neighbor_ratio",
    ]

    required_columns = ["node_id"] + feature_columns

    missing = [col for col in required_columns if col not in feature_df.columns]

    if missing:

        raise ValueError(f"Missing feature columns: {missing}")

    print("\nAligning node features...")

    feature_df = feature_df.set_index("node_id")

    feature_df = feature_df.reindex(nodes)

    if feature_df.isnull().all(axis=1).any():

        raise ValueError("Some graph nodes have no matching features")

    x = torch.tensor(feature_df[feature_columns].fillna(0).values, dtype=torch.float)

    print("Node feature tensor:", x.shape)

    print("\nBuilding edges...")

    edges = []

    for source, target in graph.edges():

        edges.append([node_mapping[source], node_mapping[target]])

        edges.append([node_mapping[target], node_mapping[source]])

    if len(edges) == 0:

        raise ValueError("Graph contains no edges")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    pyg_graph = Data(x=x, edge_index=edge_index)

    print("\nPyG Graph Created")

    print(pyg_graph)

    return pyg_graph


def main():

    print("==============================")
    print("PHASE 8 GNN DATA PREPARATION")
    print("==============================")

    graph = load_graph()

    features = load_features()

    pyg_graph = build_pyg_graph(graph, features)

    torch.save(pyg_graph, "data/graph/fraud_graph_pyg.pt")

    print("\nSaved:", "data/graph/fraud_graph_pyg.pt")

    print("==============================")
    print("GNN DATA PREPARATION COMPLETE")
    print("==============================")


if __name__ == "__main__":

    main()
