import pandas as pd

from src.features.transaction_features import create_transaction_features
from src.features.user_features import create_user_features


def build_features(input_path, output_path):

    print("Loading processed dataset...")

    df = pd.read_parquet(input_path)

    print("Input shape:", df.shape)

    print("\nCreating transaction features...")

    df = create_transaction_features(df)

    print("\nCreating user behaviour features...")

    df = create_user_features(df)

    print("\nSaving feature dataset...")

    df.to_parquet(output_path, index=False)

    print("\nFeature pipeline completed.")

    print("Final feature shape:", df.shape)


if __name__ == "__main__":

    build_features(
        "data/processed/train_processed.parquet",
        "data/processed/train_features.parquet",
    )
