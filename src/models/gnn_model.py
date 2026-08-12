"""
Graph Neural Network model for fraud detection.

Architecture:
- GraphSAGE layers
- Batch normalization
- Dropout
- Binary fraud classification
"""


import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv



class FraudGraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=2,
        dropout=0.3
    ):

        super().__init__()


        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )


        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )


        self.classifier = nn.Linear(
            hidden_dim,
            output_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )



    def forward(
        self,
        x,
        edge_index
    ):


        x = self.conv1(
            x,
            edge_index
        )


        x = torch.relu(
            x
        )


        x = self.dropout(
            x
        )


        x = self.conv2(
            x,
            edge_index
        )


        x = torch.relu(
            x
        )


        x = self.dropout(
            x
        )


        x = self.classifier(
            x
        )


        return x
