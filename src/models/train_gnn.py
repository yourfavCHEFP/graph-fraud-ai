"""
Train GraphSAGE fraud detection model.

Input:
- data/graph/fraud_graph_ready.pt

Output:
- models/graphsage_fraud_model.pt
"""


import torch
import torch.nn.functional as F

from torch_geometric.data import Data

from src.models.gnn_model import FraudGraphSAGE



def load_graph():

    print("\nLoading PyG fraud graph...")

    graph = torch.load(
        "data/graph/fraud_graph_ready.pt",
        weights_only=False
    )


    print(graph)

    return graph



def train():

    print("==============================")
    print("PHASE 10.2 GNN TRAINING")
    print("==============================")


    graph = load_graph()


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "\nDevice:",
        device
    )


    graph = graph.to(
        device
    )


    model = FraudGraphSAGE(
        input_dim=graph.x.shape[1]
    )


    model = model.to(
        device
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4
    )


    # Handle fraud imbalance

    fraud_count = graph.y.sum()

    normal_count = (
        graph.y.shape[0]
        -
        fraud_count
    )


    weight = torch.tensor(
        [
            1.0,
            normal_count / fraud_count
        ],
        dtype=torch.float
    ).to(device)



    loss_function = torch.nn.CrossEntropyLoss(
        weight=weight
    )



    epochs = 50


    print(
        "\nStarting training..."
    )


    for epoch in range(
        epochs
    ):


        model.train()


        optimizer.zero_grad()


        output = model(
            graph.x,
            graph.edge_index
        )


        loss = loss_function(
            output[
                graph.train_mask
            ],
            graph.y[
                graph.train_mask
            ]
        )


        loss.backward()


        optimizer.step()



        if epoch % 5 == 0:

            print(
                f"Epoch {epoch} | Loss {loss.item():.4f}"
            )



    torch.save(
        model.state_dict(),
        "models/graphsage_fraud_model.pt"
    )


    print(
        "\nSaved:",
        "models/graphsage_fraud_model.pt"
    )


    print("==============================")
    print("GNN TRAINING COMPLETE")
    print("==============================")



if __name__ == "__main__":

    train()
