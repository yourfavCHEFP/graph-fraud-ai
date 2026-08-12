"""
Train GraphSAGE fraud detection model.

Input:
- data/graph/fraud_graph_ready.pt

Output:
- models/graphsage_fraud_model.pt
"""


import torch

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


    graph = graph.to(device)



    model = FraudGraphSAGE(
        input_dim=graph.x.shape[1]
    )


    model = model.to(device)



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4
    )



    # ==============================
    # Handle class imbalance
    # ==============================

    fraud_count = graph.y.sum().float()


    normal_count = (
        graph.y.shape[0]
        -
        fraud_count
    )


    fraud_ratio = (
        normal_count /
        fraud_count
    )


    print(
        "\nFraud ratio:",
        fraud_ratio.item()
    )


    # Softer weighting
    fraud_weight = torch.sqrt(
        fraud_ratio
    )


    # Prevent extreme bias
    fraud_weight = torch.clamp(
        fraud_weight,
        max=5.0
    )


    print(
        "Using fraud class weight:",
        fraud_weight.item()
    )



    class_weights = torch.tensor(
        [
            1.0,
            fraud_weight.item()
        ],
        dtype=torch.float
    ).to(device)



    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )



    epochs = 50


    print(
        "\nStarting training..."
    )


    for epoch in range(epochs):


        model.train()


        optimizer.zero_grad()



        output = model(
            graph.x,
            graph.edge_index
        )



        loss = criterion(
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

            model.eval()


            with torch.no_grad():

                predictions = output.argmax(
                    dim=1
                )


                train_acc = (
                    predictions[
                        graph.train_mask
                    ]
                    ==
                    graph.y[
                        graph.train_mask
                    ]
                ).float().mean()



            print(
                f"Epoch {epoch} | "
                f"Loss {loss.item():.4f} | "
                f"Train Acc {train_acc.item():.4f}"
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
