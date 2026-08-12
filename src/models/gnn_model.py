"""
GNN architectures for fraud detection.

Supported models:
- GCN
- GAT
- GraphSAGE
"""

import torch
import torch.nn as nn

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv
)


class FraudGCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=2,
        dropout=0.3
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
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

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.dropout(x)

        return self.classifier(x)


class FraudGAT(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        output_dim=2,
        heads=2,
        dropout=0.3
    ):

        super().__init__()

        self.conv1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
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

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.dropout(x)

        return self.classifier(x)


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

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.dropout(x)

        return self.classifier(x)
