"""
Evaluation metrics for fraud detection models.
"""

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)


def evaluate_predictions(
    y_true,
    y_pred,
    y_prob
):

    metrics = {

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob
            ),

        "precision":
            precision_score(
                y_true,
                y_pred
            ),

        "recall":
            recall_score(
                y_true,
                y_pred
            ),

        "f1":
            f1_score(
                y_true,
                y_pred
            )
    }


    return metrics



def print_evaluation(
    y_true,
    y_pred
):

    print("\nClassification Report:")
    
    print(
        classification_report(
            y_true,
            y_pred
        )
    )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )
