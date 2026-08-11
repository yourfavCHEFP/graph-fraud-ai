# # This file job is
# # Read configuration
# Locate dataset files
# Load transaction data
# Load identity data
# Merge them
# Report dataset information
# Return clean dataframe objects

import os

import pandas as pd
import yaml


def load_config(path="configs/data_config.yaml"):
    """
    Load dataset configuration.
    """
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_csv(path):
    """
    Load CSV file safely.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pd.read_csv(path)


def load_ieee_dataset(config_path="configs/data_config.yaml"):
    """
    Load and merge IEEE-CIS Fraud Detection dataset.
    """

    config = load_config(config_path)

    raw_path = config["paths"]["raw_data"]

    files = config["files"]

    train_transaction_path = os.path.join(
        raw_path,
        files["train_transaction"]
    )

    train_identity_path = os.path.join(
        raw_path,
        files["train_identity"]
    )


    print("Loading transaction dataset...")
    transaction_df = load_csv(train_transaction_path)


    print("Loading identity dataset...")
    identity_df = load_csv(train_identity_path)


    print("\nTransaction shape:")
    print(transaction_df.shape)


    print("\nIdentity shape:")
    print(identity_df.shape)


    print("\nMerging datasets...")


    merged_df = transaction_df.merge(
        identity_df,
        on=config["merge"]["transaction_key"],
        how="left"
    )


    print("\nMerged dataset shape:")
    print(merged_df.shape)


    return merged_df


if __name__ == "__main__":

    df = load_ieee_dataset()

    print("\nDataset preview:")
    print(df.head())

