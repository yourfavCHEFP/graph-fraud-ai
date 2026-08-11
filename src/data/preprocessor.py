"""
Preprocessing pipeline for the IEEE-CIS Fraud Detection dataset.

Phase 3 focuses on producing a clean, reproducible dataset.
Feature engineering and graph-specific transformations are handled
in later phases.
"""

import os

import pandas as pd

from src.data.cleaner import clean_dataset


def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    missing_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Run the Phase 3 cleaning and preprocessing pipeline.
    """

    processed_df = clean_dataset(
        df=df,
        target_column=target_column,
        missing_threshold=missing_threshold,
    )

    return processed_df


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Save processed dataframe as Parquet.
    """

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    print(f"Saving processed dataset to: {output_path}")

    df.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    print("Processed dataset saved successfully.")


def load_processed_dataset(
    input_path: str,
) -> pd.DataFrame:
    """
    Load a previously processed Parquet dataset.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Processed dataset not found: {input_path}"
        )

    return pd.read_parquet(
        input_path,
        engine="pyarrow",
    )
