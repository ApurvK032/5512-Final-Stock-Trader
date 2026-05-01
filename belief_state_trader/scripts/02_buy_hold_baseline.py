"""Run the Buy and Hold baseline on the test split.

Run from the belief_state_trader folder:

    python scripts/02_buy_hold_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import backtest, baselines, data


def main():
    test = data.add_real_log_return(data.load_test())
    buy_hold_returns = baselines.buy_and_hold_returns(test["Real_Log_Return"])

    summary = backtest.summarize_backtest("Buy and Hold", buy_hold_returns)
    summary_df = pd.DataFrame([summary])

    results_path = PROJECT_ROOT / "results" / "buy_hold_summary.csv"
    summary_df.to_csv(results_path, index=False)

    print("Buy and Hold baseline on test split")
    print("-----------------------------------")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {results_path}")


if __name__ == "__main__":
    main()

