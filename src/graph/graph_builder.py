"""
NetworkX Graph Builder for IEEE-CIS Fraud Detection.

Creates a heterogeneous fraud graph:

Nodes:
- Transaction
- Card
- Email
- Device
- Address

Edges:
- Card -> Transaction
- Email -> Transaction
- Device -> Transaction
- Address -> Transaction

Output:
data/graph/fraud_graph.pkl
"""


import os
import pickle
import pandas as pd
import networkx as nx


def add_node_if_valid(
    graph,
    node_id,
    node_type,
    attributes=None
):
    """
    Add node only when identifier exists.
    """

    if pd.isna(node_id):
        return

    graph.add_node(
        node_id,
        type=node_type,
        **(attributes or {})
    )


def build_fraud_graph(
    input_path,
    output_path
):

    print("\n================================")
    print("PHASE 5 GRAPH CONSTRUCTION")
    print("================================\n")


    print("Loading feature dataset...")

    df = pd.read_parquet(
        input_path
    )


    print(
        "Dataset shape:",
        df.shape
    )


    print("\nInitializing graph...")

    graph = nx.Graph()


    print("\nCreating transaction nodes...")


    for _, row in df.iterrows():

        transaction_id = (
            f"transaction_{row['TransactionID']}"
        )

        graph.add_node(
            transaction_id,
            type="Transaction",
            isFraud=row["isFraud"],
            amount=row["TransactionAmt"]
        )


    print(
        "Transaction nodes created."
    )


    print("\nCreating entity relationships...")


    for _, row in df.iterrows():

        transaction_node = (
            f"transaction_{row['TransactionID']}"
        )


        # Card relationship

        if not pd.isna(row.get("card1")):

            card_node = (
                f"card_{row['card1']}"
            )

            add_node_if_valid(
                graph,
                card_node,
                "Card"
            )

            graph.add_edge(
                card_node,
                transaction_node,
                relation="card_transaction"
            )


        # Email relationship

        email = row.get(
            "P_emaildomain"
        )

        if not pd.isna(email):

            email_node = (
                f"email_{email}"
            )

            add_node_if_valid(
                graph,
                email_node,
                "Email"
            )

            graph.add_edge(
                email_node,
                transaction_node,
                relation="email_transaction"
            )


        # Device relationship

        device = row.get(
            "DeviceInfo"
        )

        if not pd.isna(device):

            device_node = (
                f"device_{device}"
            )

            add_node_if_valid(
                graph,
                device_node,
                "Device"
            )

            graph.add_edge(
                device_node,
                transaction_node,
                relation="device_transaction"
            )


        # Address relationship

        address = row.get(
            "addr1"
        )

        if not pd.isna(address):

            address_node = (
                f"address_{address}"
            )

            add_node_if_valid(
                graph,
                address_node,
                "Address"
            )

            graph.add_edge(
                address_node,
                transaction_node,
                relation="address_transaction"
            )


    print("\nGraph construction completed.")


    print(
        "Nodes:",
        graph.number_of_nodes()
    )

    print(
        "Edges:",
        graph.number_of_edges()
    )


    print("\nSaving graph...")


    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )


    with open(
        output_path,
        "wb"
    ) as f:

        pickle.dump(
            graph,
            f
        )


    print(
        "\nGraph saved:",
        output_path
    )


    print("\n================================")
    print("GRAPH BUILD COMPLETE")
    print("================================")


if __name__ == "__main__":

    build_fraud_graph(
        "data/processed/train_features.parquet",
        "data/graph/fraud_graph.pkl"
    )
