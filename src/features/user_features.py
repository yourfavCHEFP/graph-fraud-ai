import pandas as pd


def create_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioural features based on card usage.
    """

    print("Creating card behaviour features...")

    card_columns = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
    ]

    available_cards = [col for col in card_columns if col in df.columns]

    if not available_cards:
        return df

    card_key = "card1"

    if card_key in df.columns:

        card_stats = (
            df.groupby(card_key, observed=False)
            .agg(
                card_transaction_count=("TransactionID", "count"),
                card_average_amount=("TransactionAmt", "mean"),
                card_total_amount=("TransactionAmt", "sum"),
            )
            .reset_index()
        )

        df = df.merge(card_stats, on=card_key, how="left")

    return df


def create_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create email behaviour features.
    """

    print("Creating email behaviour features...")

    email_columns = [
        "P_emaildomain",
        "R_emaildomain",
    ]

    for email_col in email_columns:

        if email_col in df.columns:

            email_stats = (
                df.groupby(email_col, observed=False)
                .agg(
                    email_transaction_count=("TransactionID", "count"),
                    email_average_amount=("TransactionAmt", "mean"),
                )
                .reset_index()
            )

            df = df.merge(email_stats, on=email_col, how="left")

    return df


def create_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create device behaviour features.
    """

    print("Creating device behaviour features...")

    if "DeviceInfo" in df.columns:

        device_stats = (
            df.groupby("DeviceInfo", observed=False)
            .agg(device_transaction_count=("TransactionID", "count"))
            .reset_index()
        )

        df = df.merge(device_stats, on="DeviceInfo", how="left")

    return df


def create_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete user behavioural feature pipeline.
    """

    print("\n========== USER FEATURES ==========")

    df = create_card_features(df)

    df = create_email_features(df)

    df = create_device_features(df)

    print("User behavioural features created.")

    print("Current shape:", df.shape)

    print("===================================\n")

    return df
