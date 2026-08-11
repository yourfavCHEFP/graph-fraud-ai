"""
Run the Phase 3 IEEE-CIS data cleaning and preprocessing pipeline.
"""

import os
import sys

import yaml

# Ensure project root is available when this script is executed directly.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_ieee_dataset
from src.data.preprocessor import (
    preprocess_dataset,
    save_processed_dataset,
)


CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "data_config.yaml",
)


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    target_column = config["target"]["column"]

    missing_threshold = config["processing"][
        "missing_threshold"
    ]

    processed_directory = os.path.join(
        PROJECT_ROOT,
        config["paths"]["processed_data"],
    )

    output_file = config["processing"][
        "processed_train_file"
    ]

    output_path = os.path.join(
        processed_directory,
        output_file,
    )

    print("\n==============================================")
    print("IEEE-CIS PHASE 3 PREPROCESSING PIPELINE")
    print("==============================================")

    print("\n[1/3] Loading raw IEEE-CIS dataset...")

    df = load_ieee_dataset(
        os.path.join(
            PROJECT_ROOT,
            "configs",
            "data_config.yaml",
        )
    )

    print(f"Raw dataset shape: {df.shape}")

    print("\n[2/3] Cleaning and preprocessing dataset...")

    processed_df = preprocess_dataset(
        df=df,
        target_column=target_column,
        missing_threshold=missing_threshold,
    )

    print("\n[3/3] Saving processed dataset...")

    save_processed_dataset(
        df=processed_df,
        output_path=output_path,
    )

    print("\n==============================================")
    print("PHASE 3 PREPROCESSING COMPLETE")
    print("==============================================")

    print(f"Processed shape: {processed_df.shape}")
    print(f"Output: {output_path}")

    print("\nTarget distribution:")

    print(
        processed_df[target_column]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()
