"""
IEEE-CIS Fraud Detection dataset loader.

Responsibilities:
    - Read configuration.
    - Locate dataset files.
    - Load transaction data.
    - Load identity data.
    - Merge transaction and identity data.
    - Report dataset information.
    - Return the merged dataframe.
"""

import os

import pandas as pd
import yaml

# ============================================================
# LOAD CONFIGURATION
# ============================================================


def load_config(
    path="configs/data_config.yaml",
):
    """
    Load dataset configuration from YAML.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(
        path,
        "r",
    ) as file:

        return yaml.safe_load(file)


# ============================================================
# LOAD CSV
# ============================================================


def load_csv(path):
    """
    Load a CSV file safely.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pd.read_csv(path)


# ============================================================
# LOAD IEEE-CIS DATASET
# ============================================================


def load_ieee_dataset(
    config_path="configs/data_config.yaml",
):
    """
    Load and merge the IEEE-CIS Fraud Detection dataset.
    """

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../..",
        )
    )

    # --------------------------------------------------------
    # Resolve configuration path
    # --------------------------------------------------------

    if not os.path.isabs(config_path):

        config_path = os.path.join(
            project_root,
            config_path,
        )

    config = load_config(config_path)

    # --------------------------------------------------------
    # Resolve raw data directory
    # --------------------------------------------------------

    raw_path = os.path.join(
        project_root,
        config["paths"]["raw_data"],
    )

    files = config["files"]

    transaction_file = files["train_transaction"]

    identity_file = files["train_identity"]

    transaction_path = os.path.join(
        raw_path,
        transaction_file,
    )

    identity_path = os.path.join(
        raw_path,
        identity_file,
    )

    # ========================================================
    # LOAD TRANSACTION DATA
    # ========================================================

    print("Loading transaction dataset...")

    transaction_df = load_csv(transaction_path)

    # ========================================================
    # LOAD IDENTITY DATA
    # ========================================================

    print("Loading identity dataset...")

    identity_df = load_csv(identity_path)

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    print("\nTransaction shape:")

    print(transaction_df.shape)

    print("\nIdentity shape:")

    print(identity_df.shape)

    # ========================================================
    # MERGE
    # ========================================================

    merge_key = config["merge"]["transaction_key"]

    if merge_key not in transaction_df.columns:
        raise KeyError(f"Merge key '{merge_key}' " "not found in transaction dataset.")

    if merge_key not in identity_df.columns:
        raise KeyError(f"Merge key '{merge_key}' " "not found in identity dataset.")

    print("\nMerging datasets...")

    merged_df = transaction_df.merge(
        identity_df,
        on=merge_key,
        how="left",
    )

    print("\nMerged dataset shape:")

    print(merged_df.shape)

    return merged_df


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    df = load_ieee_dataset()

    print("\nDataset preview:")

    print(df.head())
