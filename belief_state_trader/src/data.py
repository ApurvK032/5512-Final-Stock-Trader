"""Data loading helpers for the belief-state trader pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_CSV = DATA_DIR / "sp500_train.csv"
TEST_CSV = DATA_DIR / "sp500_test.csv"

FEATURE_COLUMNS = ["Log_Return", "Volume", "Volatility"]


def load_csv(path: Path) -> pd.DataFrame:
    """Read one S&P 500 CSV, parse dates, sort rows, and use Date as index."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df


def load_train() -> pd.DataFrame:
    """Load the training split."""
    return load_csv(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    """Load the test split."""
    return load_csv(TEST_CSV)


def feature_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return the model feature columns from a loaded dataframe."""
    columns = FEATURE_COLUMNS if columns is None else columns
    return df[columns].copy()


def add_real_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add real daily log returns computed from Adj_Close."""
    out = df.copy()
    out["Real_Log_Return"] = np.log(out["Adj_Close"] / out["Adj_Close"].shift(1))
    return out
