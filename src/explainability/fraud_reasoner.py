"""Human readable fraud reasoning layer."""

from src.explainability.model_explainer import explain_prediction
from src.explainability.graph_explainer import analyze_transaction_neighborhood


def build_fraud_explanation(adjacency_index, num_nodes, transaction_id, probability, threshold):
    """
    FIX (mentor review item 11): signature changed from
    build_fraud_explanation(graph, transaction_id, ...) to take the
    precomputed adjacency_index + num_nodes instead of the full graph
    object -- see src/explainability/graph_explainer.py's docstring.
    This also means the full graph no longer needs to stay in memory
    just to answer "who are this transaction's neighbors" (see
    src/inference/predictor.py, which now frees self.graph after
    building the adjacency index once at startup).
    """
    return {
        **explain_prediction(probability, threshold),
        "graph_context": analyze_transaction_neighborhood(
            adjacency_index,
            num_nodes,
            transaction_id,
        ),
    }
