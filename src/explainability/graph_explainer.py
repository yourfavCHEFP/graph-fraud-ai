"""Phase 15 graph investigation utilities."""

import torch


def analyze_transaction_neighborhood(graph, transaction_id, limit=5):
    if transaction_id >= graph.x.shape[0]:
        raise ValueError("transaction_id outside graph range")

    node_edges = graph.edge_index[:, graph.edge_index[0] == transaction_id]
    neighbors = node_edges[1].tolist()[:limit]

    return {
        "transaction_id": int(transaction_id),
        "neighbor_count": int(len(neighbors)),
        "neighbors": [int(x) for x in neighbors],
    }
