"""Run the Moving Average Crossover baseline."""
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
    strategy_returns = baselines.moving_average_crossover_returns(
        prices=test["Adj_Close"],
        real_log_returns=test["Real_Log_Return"],
        short_window=50,
        long_window=200,
    )

    summary = backtest.summarize_backtest("Moving Average Crossover 50/200", strategy_returns)
    summary_df = pd.DataFrame([summary])

    results_path = PROJECT_ROOT / "results" / "moving_average_summary.csv"
    summary_df.to_csv(results_path, index=False)

    print("Moving Average Crossover baseline on test split")
    print("------------------------------------------------")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {results_path}")


if __name__ == "__main__":
    main()

