"""
Graph Feature Engineering for IEEE-CIS Fraud Detection.

Creates numerical graph-based features from the fraud graph.

Features generated:

Transaction nodes:
- transaction_degree
- fraud_neighbor_count
- fraud_neighbor_ratio

Entity nodes:
- entity_degree

Output:
data/graph/node_features.parquet
"""


import os
import pickle
import pandas as pd
import networkx as nx



def load_graph(
    graph_path
):

    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Graph not found: {graph_path}"
        )

    print("Loading fraud graph...")

    with open(
        graph_path,
        "rb"
    ) as file:

        graph = pickle.load(file)

    print(
        "Graph loaded."
    )

    print(
        "Nodes:",
        graph.number_of_nodes()
    )

    print(
        "Edges:",
        graph.number_of_edges()
    )

    return graph



def create_graph_features(
    graph
):

    print("\nCreating graph features...")


    features = []


    for node, data in graph.nodes(
        data=True
    ):


        node_type = data.get(
            "type"
        )


        degree = graph.degree(
            node
        )


        row = {

            "node_id": node,

            "node_type": node_type,

            "degree": degree

        }


        #
        # Transaction specific features
        #

        if node_type == "Transaction":


            fraud_neighbors = 0

            total_neighbors = 0


            for neighbor in graph.neighbors(
                node
            ):

                neighbor_data = graph.nodes[
                    neighbor
                ]


                if (
                    neighbor_data.get("type")
                    ==
                    "Transaction"
                ):

                    continue


                total_neighbors += 1


                if (
                    neighbor_data.get("isFraud")
                    == 1
                ):

                    fraud_neighbors += 1



            row.update({

                "transaction_entity_count":
                    total_neighbors,


                "fraud_neighbor_count":
                    fraud_neighbors,


                "fraud_neighbor_ratio":
                    (
                        fraud_neighbors /
                        total_neighbors
                        if total_neighbors > 0
                        else 0
                    )

            })


        features.append(
            row
        )



    feature_df = pd.DataFrame(
        features
    )


    print(
        "Graph feature shape:",
        feature_df.shape
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


    graph_path = (
        "data/graph/fraud_graph.pkl"
    )


    output_path = (
        "data/graph/node_features.parquet"
    )


    print("\n==============================")
    print("PHASE 6 GRAPH FEATURES")
    print("==============================\n")


    graph = load_graph(
        graph_path
    )


    feature_df = create_graph_features(
        graph
    )


    save_graph_features(
        feature_df,
        output_path
    )


    print("\n==============================")
    print("GRAPH FEATURE ENGINEERING COMPLETE")
    print("==============================")



if __name__ == "__main__":

    main()
