"""Human readable fraud reasoning layer."""

from src.explainability.model_explainer import explain_prediction
from src.explainability.graph_explainer import analyze_transaction_neighborhood


def build_fraud_explanation(graph, transaction_id, probability, threshold):
    return {
        **explain_prediction(probability, threshold),
        "graph_context": analyze_transaction_neighborhood(
            graph,
            transaction_id,
        ),
    }
