"""
XGBoost baseline fraud detection model.

Input:
    data/processed/train_graph_features.parquet

Output:
    models/xgboost_baseline.json
"""


import os
import json

import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


RANDOM_STATE = 42


DATA_PATH = (
    "data/processed/train_graph_features.parquet"
)


MODEL_PATH = (
    "models/xgboost_baseline.json"
)



def load_dataset(path):

    print("\nLoading dataset...")

    df = pd.read_parquet(path)

    print(
        "Dataset shape:",
        df.shape
    )

    return df



def prepare_data(df):

    print("\nPreparing features...")


    target = "isFraud"


    X = df.drop(
        columns=[
            target,
            "TransactionID"
        ]
    )


    y = df[target]


    print(
        "Feature shape:",
        X.shape
    )


    print(
        "Fraud ratio:",
        y.mean()
    )


    # Convert pandas category columns
    category_columns = (
        X.select_dtypes(
            include="category"
        )
        .columns
        .tolist()
    )


    print(
        "Categorical features:",
        len(category_columns)
    )


    for col in category_columns:

        X[col] = (
            X[col]
            .cat.codes
        )


    return X, y



def train_model(
    X_train,
    y_train
):

    print("\nTraining XGBoost model...")


    fraud_ratio = (
        y_train.value_counts()[0]
        /
        y_train.value_counts()[1]
    )


    model = xgb.XGBClassifier(

        n_estimators=500,

        max_depth=7,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="auc",

        tree_method="hist",

        scale_pos_weight=fraud_ratio,

        random_state=RANDOM_STATE,

        n_jobs=-1

    )


    model.fit(
        X_train,
        y_train
    )


    print(
        "Training complete."
    )


    return model



def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\nEvaluating model...")


    probabilities = (
        model.predict_proba(
            X_test
        )[:,1]
    )


    predictions = (
        probabilities >= 0.5
    )


    metrics = {

        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            ),

        "precision":
            precision_score(
                y_test,
                predictions
            ),

        "recall":
            recall_score(
                y_test,
                predictions
            ),

        "f1":
            f1_score(
                y_test,
                predictions
            )

    }


    print(
        json.dumps(
            metrics,
            indent=4
        )
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    return metrics



def save_model(model):

    print("\nSaving model...")


    os.makedirs(
        "models",
        exist_ok=True
    )


    model.save_model(
        MODEL_PATH
    )


    print(
        "Saved:",
        MODEL_PATH
    )



def main():

    print("\n==============================")
    print("PHASE 7 XGBOOST BASELINE")
    print("==============================\n")


    df = load_dataset(
        DATA_PATH
    )


    X, y = prepare_data(
        df
    )


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        stratify=y,

        random_state=RANDOM_STATE

    )


    print(
        "\nTrain shape:",
        X_train.shape
    )


    print(
        "Test shape:",
        X_test.shape
    )


    model = train_model(
        X_train,
        y_train
    )


    evaluate_model(
        model,
        X_test,
        y_test
    )


    save_model(
        model
    )


    print("\n==============================")
    print("BASELINE TRAINING COMPLETE")
    print("==============================")



if __name__ == "__main__":

    main()
