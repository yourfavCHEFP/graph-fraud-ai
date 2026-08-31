"""Phase 15 graph investigation utilities.

FIX (mentor review item 11): analyze_transaction_neighborhood() used to
scan graph.edge_index[0] == transaction_id -- a full pass over EVERY edge
in the graph, on EVERY prediction request. For a 606K-node graph with
millions of edges, that's real, avoidable per-request latency and CPU
work for a lookup that never changes between requests (the graph is
frozen after startup).

build_adjacency_index() now does that scan ONCE, when the graph is
loaded (see src/inference/predictor.py), producing a plain dict for O(1)
neighbor lookups per request afterward.
"""

from collections import defaultdict


def build_adjacency_index(graph, limit=5):
    """
    Builds {source_node_id: [neighbor_ids...]} once, capped at `limit`
    neighbors per node (matching the UI's five-neighbor display limit --
    see analyze_transaction_neighborhood's docstring for what these IDs
    represent).
    """
    index = defaultdict(list)
    sources = graph.edge_index[0].tolist()
    targets = graph.edge_index[1].tolist()

    for src, dst in zip(sources, targets):
        if len(index[src]) < limit:
            index[src].append(dst)

    return dict(index)


def analyze_transaction_neighborhood(adjacency_index, num_nodes, transaction_id, limit=5):
    """
    Returns the transaction node's neighbors from the precomputed
    adjacency index (O(1) dict lookup instead of an O(edges) tensor scan).

    NOTE ON WHAT THESE IDS REPRESENT: this graph is heterogeneous
    (transaction, card, email, device, and address nodes all share one
    node-ID space per src/graph/pyg_converter.py). The returned neighbor
    IDs are NOT guaranteed to be transaction nodes -- they may be any
    adjacent entity node (e.g. a shared card or device). The node's type
    can be recovered from its `node_type_id` feature column if needed.
    """
    if transaction_id >= num_nodes:
        raise ValueError("transaction_id outside graph range")

    neighbors = adjacency_index.get(transaction_id, [])[:limit]

    return {
        "transaction_id": int(transaction_id),
        "neighbor_count": int(len(neighbors)),
        "neighbors": [int(x) for x in neighbors],
    }
