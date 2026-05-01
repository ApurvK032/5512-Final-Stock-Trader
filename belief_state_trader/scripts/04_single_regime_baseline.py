"""Run the Single-Regime No-Belief baseline on the test split.

Run from the belief_state_trader folder:

    python scripts/04_single_regime_baseline.py
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
    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    strategy_returns, weight = baselines.single_regime_no_belief_returns(
        train_real_log_returns=train["Real_Log_Return"],
        test_real_log_returns=test["Real_Log_Return"],
        risk_aversion=2.0,
    )

    summary = backtest.summarize_backtest("Single-Regime No-Belief", strategy_returns)
    summary["fixed_weight"] = weight
    summary_df = pd.DataFrame([summary])

    results_path = PROJECT_ROOT / "results" / "single_regime_summary.csv"
    summary_df.to_csv(results_path, index=False)

    print("Single-Regime No-Belief baseline on test split")
    print("-----------------------------------------------")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {results_path}")


if __name__ == "__main__":
    main()
