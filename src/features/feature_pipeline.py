import pandas as pd

from src.features.transaction_features import (
    create_transaction_features
)


def build_features(
    input_path,
    output_path
):

    print("Loading processed dataset...")

    df = pd.read_parquet(
        input_path
    )


    print(
        "Input shape:",
        df.shape
    )


    df = create_transaction_features(
        df
    )


    print(
        "Saving feature dataset..."
    )


    df.to_parquet(
        output_path,
        index=False
    )


    print(
        "Feature pipeline completed."
    )


if __name__ == "__main__":


    build_features(
        "data/processed/train_processed.parquet",
        "data/processed/train_features.parquet"
    )
