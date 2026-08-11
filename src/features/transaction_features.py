import numpy as np
import pandas as pd


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction-level fraud features.
    """

    df = df.copy()

    print("Creating transaction features...")

    # -----------------------------
    # Transaction Amount Features
    # -----------------------------

    if "TransactionAmt" in df.columns:

        df["TransactionAmt_log"] = np.log1p(
            df["TransactionAmt"]
        )

        df["TransactionAmt_decimal"] = (
            df["TransactionAmt"] % 1
        )

        df["TransactionAmt_is_round"] = (
            df["TransactionAmt_decimal"] == 0
        ).astype(int)


    # -----------------------------
    # Time Features
    # -----------------------------

    if "TransactionDT" in df.columns:

        seconds = df["TransactionDT"]

        df["transaction_hour"] = (
            (seconds // 3600) % 24
        )

        df["transaction_day"] = (
            seconds // 86400
        )

        df["transaction_week"] = (
            seconds // (86400 * 7)
        )


    # -----------------------------
    # Missingness Features
    # -----------------------------

    df["missing_total_count"] = (
        df.isna().sum(axis=1)
    )

    df["missing_percentage"] = (
        df["missing_total_count"]
        /
        df.shape[1]
    )


    print(
        f"Transaction features created. "
        f"Shape: {df.shape}"
    )


    return df
