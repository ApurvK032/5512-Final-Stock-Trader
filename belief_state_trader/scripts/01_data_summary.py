"""Print a basic summary of the S&P 500 train/test data.

Run from the belief_state_trader folder:

    python scripts/01_data_summary.py
"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data


def print_split_summary(name: str, df):
    """Print basic shape, date, missing-value, and feature-stat information."""
    with_returns = data.add_real_log_return(df)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")

    print("\nmissing values:")
    print(df.isna().sum().to_string())

    print("\nfeature statistics:")
    print(df[data.FEATURE_COLUMNS].describe().round(4).to_string())

    print("\nreal return check:")
    real_returns = with_returns["Real_Log_Return"].dropna()
    print(f"real return rows after first-day drop: {len(real_returns)}")
    print(real_returns.describe().round(6).to_string())


def main():
    train = data.load_train()
    test = data.load_test()

    print_split_summary("Training split", train)
    print_split_summary("Test split", test)


if __name__ == "__main__":
    main()
