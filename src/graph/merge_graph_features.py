"""
Merge graph-derived features back into transaction dataset.

Input:
- data/processed/train_features.parquet
- data/graph/node_features.parquet

Output:
- data/processed/train_graph_features.parquet
"""

import os

import pandas as pd


def load_transaction_features(path):

    print("Loading transaction features...")

    df = pd.read_parquet(path)

    print("Transaction feature shape:", df.shape)

    return df


def load_graph_features(path):

    print("\nLoading graph features...")

    graph_df = pd.read_parquet(path)

    print("Graph feature shape:", graph_df.shape)

    return graph_df


def prepare_transaction_graph_features(graph_df):

    print("\nFiltering transaction graph nodes...")

    transaction_graph = graph_df[graph_df["node_type"] == "Transaction"].copy()

    print("Transaction graph nodes:", transaction_graph.shape)

    print("\nExtracting TransactionID...")

    transaction_graph["TransactionID"] = (
        transaction_graph["node_id"]
        .str.replace("transaction_", "", regex=False)
        .astype("int64")
    )

    transaction_graph = transaction_graph.drop(columns=["node_id", "node_type"])

    return transaction_graph


def merge_features(transaction_df, graph_df):

    print("\nMerging graph features...")

    merged = transaction_df.merge(graph_df, on="TransactionID", how="left")

    print("Merged shape:", merged.shape)

    return merged


def main():

    print("\n==============================")
    print("PHASE 6.2 GRAPH FEATURE MERGE")
    print("==============================\n")

    transaction_path = "data/processed/train_features.parquet"

    graph_path = "data/graph/node_features.parquet"

    output_path = "data/processed/train_graph_features.parquet"

    transaction_df = load_transaction_features(transaction_path)

    graph_df = load_graph_features(graph_path)

    graph_df = prepare_transaction_graph_features(graph_df)

    final_df = merge_features(transaction_df, graph_df)

    print("\nSaving final dataset...")

    os.makedirs("data/processed", exist_ok=True)

    final_df.to_parquet(output_path, index=False)

    print("\nSaved:", output_path)

    print("\n==============================")
    print("GRAPH FEATURE MERGE COMPLETE")
    print("==============================")


if __name__ == "__main__":

    main()
