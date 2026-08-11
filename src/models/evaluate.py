"""
Model evaluation pipeline.
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.models.metrics import evaluate_predictions, print_evaluation


def main():

    print("==============================")
    print("PHASE 7.2 MODEL EVALUATION")
    print("==============================")

    print("\nLoading dataset...")

    df = pd.read_parquet("data/processed/train_graph_features.parquet")

    print("Dataset shape:", df.shape)

    print("\nPreparing features...")

    drop_columns = ["isFraud", "TransactionID"]

    X = df.drop(columns=[c for c in drop_columns if c in df.columns])

    y = df["isFraud"]

    print("Feature shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nLoading model...")

    model = xgb.XGBClassifier()

    model.load_model("models/xgboost_baseline.json")

    print("\nChecking feature alignment...")

    trained_features = model.get_booster().feature_names

    if trained_features:

        missing = set(trained_features) - set(X_test.columns)
        extra = set(X_test.columns) - set(trained_features)

        if missing:
            raise ValueError(f"Missing features: {missing}")

        if extra:
            print("Removing extra features:", extra)

            X_test = X_test[trained_features]

    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    results = evaluate_predictions(y_test, predictions, probabilities)

    print("\nMetrics:")

    for key, value in results.items():

        print(f"{key}: {value:.4f}")

    print_evaluation(y_test, predictions)

    print("\n==============================")
    print("MODEL EVALUATION COMPLETE")
    print("==============================")


if __name__ == "__main__":

    main()
