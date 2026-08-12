"""
Train GNN fraud detection models.

Models:
- GCN
- GAT
- GraphSAGE

Input:
- data/graph/fraud_graph_ready.pt

Output:
- models/{model_name}_best.pt
"""


import os
import torch

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

from src.models.gnn_model import (
    FraudGCN,
    FraudGAT,
    FraudGraphSAGE
)



MODEL_DIR = "models"


def load_graph():

    print("\nLoading fraud graph...")

    graph = torch.load(
        "data/graph/fraud_graph_ready.pt",
        weights_only=False
    )

    print(graph)

    return graph



def normalize_features(graph):

    print("\nNormalizing node features...")


    train_features = graph.x[
        graph.train_mask
    ]


    mean = train_features.mean(
        dim=0
    )


    std = train_features.std(
        dim=0
    )


    std[std == 0] = 1


    graph.x = (
        graph.x - mean
    ) / std


    return graph



def calculate_class_weights(graph, device):

    print("\nCalculating class weights...")


    labels = graph.y[
        graph.train_mask
    ]


    fraud = (
        labels == 1
    ).sum().float()


    normal = (
        labels == 0
    ).sum().float()


    ratio = normal / fraud


    weight = torch.tensor(
        [
            1.0,
            min(torch.sqrt(ratio), torch.tensor(5.0))
        ],
        dtype=torch.float
    ).to(device)


    print(
        "Fraud weight:",
        weight[1].item()
    )


    return weight



def get_models(input_dim):

    return {

        "gcn":
            FraudGCN(
                input_dim
            ),


        "gat":
            FraudGAT(
                input_dim
            ),


        "graphsage":
            FraudGraphSAGE(
                input_dim
            )
    }



def evaluate_validation(
    model,
    graph,
    device
):

    model.eval()


    with torch.no_grad():

        output = model(
            graph.x,
            graph.edge_index
        )


        probability = torch.softmax(
            output,
            dim=1
        )[:,1]


    y_true = graph.y[
        graph.val_mask
    ].cpu().numpy()


    y_prob = probability[
        graph.val_mask
    ].cpu().numpy()


    roc = roc_auc_score(
        y_true,
        y_prob
    )


    pr = average_precision_score(
        y_true,
        y_prob
    )


    return roc, pr



def train_model(
    name,
    model,
    graph,
    device,
    criterion
):


    print("\n==============================")
    print(
        f"TRAINING {name.upper()}"
    )
    print("==============================")


    model = model.to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4
    )


    best_pr = 0


    epochs = 50


    for epoch in range(
        epochs
    ):


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



        roc, pr = evaluate_validation(
            model,
            graph,
            device
        )



        if pr > best_pr:


            best_pr = pr


            torch.save(
                model.state_dict(),
                f"{MODEL_DIR}/{name}_best.pt"
            )



        if epoch % 5 == 0:

            print(
                f"Epoch {epoch} | "
                f"Loss {loss.item():.4f} | "
                f"Val ROC {roc:.4f} | "
                f"Val PR {pr:.4f}"
            )



    print(
        f"\nBest validation PR-AUC: {best_pr:.4f}"
    )



def main():

    print("==============================")
    print("PHASE 10 GNN MODEL TRAINING")
    print("==============================")


    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "\nDevice:",
        device
    )


    graph = load_graph()


    graph = normalize_features(
        graph
    )


    graph = graph.to(
        device
    )


    weights = calculate_class_weights(
        graph,
        device
    )


    criterion = torch.nn.CrossEntropyLoss(
        weight=weights
    )


    models = get_models(
        graph.x.shape[1]
    )


    for name, model in models.items():

        train_model(
            name,
            model,
            graph,
            device,
            criterion
        )


    print("\n==============================")
    print("PHASE 10 TRAINING COMPLETE")
    print("==============================")



if __name__ == "__main__":

    main()
