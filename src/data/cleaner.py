"""
Data cleaning utilities for the IEEE-CIS Fraud Detection dataset.
"""

import numpy as np
import pandas as pd


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce dataframe memory usage through numeric downcasting
    and categorical conversion where appropriate.
    """

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

        elif pd.api.types.is_object_dtype(col_type):
            # Convert low-cardinality string columns to category.
            # High-cardinality columns are left as object to avoid
            # creating unnecessarily large category dictionaries.
            nunique = df[col].nunique(dropna=False)
            total = len(df)

            if total > 0 and (nunique / total) < 0.5:
                df[col] = df[col].astype("category")

    return df


def remove_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 0.90,
    protected_columns=None,
) -> pd.DataFrame:
    """
    Remove columns whose missing-value ratio exceeds the threshold.

    Protected columns are never removed.
    """

    if protected_columns is None:
        protected_columns = []

    missing_ratio = df.isna().mean()

    columns_to_drop = [
        column
        for column in df.columns
        if missing_ratio[column] > threshold and column not in protected_columns
    ]

    if columns_to_drop:
        print(
            f"Removing {len(columns_to_drop)} columns "
            f"with more than {threshold:.0%} missing values."
        )

    return df.drop(columns=columns_to_drop)


def fill_missing_values(
    df: pd.DataFrame,
    categorical_fill_value: str = "Unknown",
) -> pd.DataFrame:
    """
    Fill remaining missing values.

    Numeric columns:
        Median imputation.

    Categorical/object columns:
        'Unknown'.
    """

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in numeric_columns:
        if df[column].isna().any():
            median_value = df[column].median()

            if pd.isna(median_value):
                median_value = 0

            df[column] = df[column].fillna(median_value)

    for column in categorical_columns:
        if df[column].isna().any():
            if isinstance(df[column].dtype, pd.CategoricalDtype):
                if categorical_fill_value not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([categorical_fill_value])

            df[column] = df[column].fillna(categorical_fill_value)

    return df


def validate_target(
    df: pd.DataFrame,
    target_column: str = "isFraud",
) -> pd.DataFrame:
    """
    Validate the fraud target column.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in dataset.")

    if df[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")

    unique_values = sorted(df[target_column].unique().tolist())

    if not set(unique_values).issubset({0, 1}):
        raise ValueError(f"Unexpected target values found: {unique_values}")

    df[target_column] = df[target_column].astype("int8")

    return df


def clean_dataset(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    missing_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Execute the complete Phase 3 cleaning pipeline.
    """

    print("\n========== DATA CLEANING ==========")
    print(f"Initial shape: {df.shape}")

    # Remove exact duplicate rows.
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing {duplicate_count:,} duplicate rows.")
        df = df.drop_duplicates()

    # TransactionID must be retained.
    protected_columns = [
        target_column,
        "TransactionID",
    ]

    # Remove extremely sparse columns.
    df = remove_high_missing_columns(
        df,
        threshold=missing_threshold,
        protected_columns=protected_columns,
    )

    # Validate target before further processing.
    df = validate_target(
        df,
        target_column=target_column,
    )

    # Fill remaining missing values.
    df = fill_missing_values(df)

    # Optimize memory.
    df = optimize_memory(df)

    print(f"Final cleaned shape: {df.shape}")
    print(f"Remaining missing values: " f"{df.isna().sum().sum():,}")

    print("===================================\n")

    return df
