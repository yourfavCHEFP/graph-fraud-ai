"""
Leakage-safe graph feature engineering for IEEE-CIS Fraud Detection.

Creates numerical features for transaction and entity nodes.

Transaction features:
- degree
- transaction_entity_count
- log_transaction_amount
- card_degree
- email_degree
- device_degree
- address_degree

Entity features:
- degree

No fraud labels are used as input features.

Output:
data/graph/node_features.parquet
"""

import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx


NODE_TYPE_MAP = {
    "Transaction": 0,
    "Card": 1,
    "Email": 2,
    "Device": 3,
    "Address": 4,
}


def load_graph(graph_path):

    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Graph not found: {graph_path}"
        )

    print("Loading fraud graph...")

    with open(graph_path, "rb") as file:
        graph = pickle.load(file)

    print("Graph loaded.")
    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())

    return graph


def create_graph_features(graph):

    print("\nCreating leakage-safe graph features...")

    features = []

    for node, data in graph.nodes(data=True):

        node_type = data.get("type")

        degree = graph.degree(node)

        row = {
            "node_id": node,
            "node_type": node_type,
            "node_type_id": NODE_TYPE_MAP.get(
                node_type,
                -1
            ),
            "degree": degree,
            "transaction_entity_count": 0,
            "log_transaction_amount": 0.0,
            "card_degree": 0,
            "email_degree": 0,
            "device_degree": 0,
            "address_degree": 0,
        }

        if node_type == "Transaction":

            neighbors = list(
                graph.neighbors(node)
            )

            row["transaction_entity_count"] = len(
                neighbors
            )

            amount = data.get(
                "amount",
                0
            )

            if pd.isna(amount):
                amount = 0

            row["log_transaction_amount"] = np.log1p(
                max(float(amount), 0.0)
            )

            for neighbor in neighbors:

                neighbor_data = graph.nodes[
                    neighbor
                ]

                neighbor_type = neighbor_data.get(
                    "type"
                )

                neighbor_degree = graph.degree(
                    neighbor
                )

                if neighbor_type == "Card":
                    row["card_degree"] = max(
                        row["card_degree"],
                        neighbor_degree
                    )

                elif neighbor_type == "Email":
                    row["email_degree"] = max(
                        row["email_degree"],
                        neighbor_degree
                    )

                elif neighbor_type == "Device":
                    row["device_degree"] = max(
                        row["device_degree"],
                        neighbor_degree
                    )

                elif neighbor_type == "Address":
                    row["address_degree"] = max(
                        row["address_degree"],
                        neighbor_degree
                    )

        features.append(row)

    feature_df = pd.DataFrame(
        features
    )

    print(
        "Graph feature shape:",
        feature_df.shape
    )

    print(
        "\nFeature columns:"
    )

    for column in feature_df.columns:
        print(
            f"- {column}"
        )

    print(
        "\nNode type distribution:"
    )

    print(
        feature_df["node_type"].value_counts()
    )

    print(
        "\nChecking numerical features..."
    )

    numerical_columns = [
        column
        for column in feature_df.columns
        if column not in [
            "node_id",
            "node_type"
        ]
    ]

    if feature_df[
        numerical_columns
    ].isna().any().any():

        raise ValueError(
            "NaN values detected in graph features."
        )

    if np.isinf(
        feature_df[
            numerical_columns
        ].to_numpy()
    ).any():

        raise ValueError(
            "Infinite values detected in graph features."
        )

    print(
        "Feature validation passed."
    )

    return feature_df


def save_graph_features(
    df,
    output_path
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    print(
        "\nSaving graph features..."
    )

    df.to_parquet(
        output_path,
        index=False
    )

    print(
        "Saved:",
        output_path
    )


def main():

    print(
        "\n=============================="
    )

    print(
        "PHASE 6 GRAPH FEATURES"
    )

    print(
        "==============================\n"
    )

    graph = load_graph(
        "data/graph/fraud_graph.pkl"
    )

    feature_df = create_graph_features(
        graph
    )

    save_graph_features(
        feature_df,
        "data/graph/node_features.parquet"
    )

    print(
        "\n=============================="
    )

    print(
        "GRAPH FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()
