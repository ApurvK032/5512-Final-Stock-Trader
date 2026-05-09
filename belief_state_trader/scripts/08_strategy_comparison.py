"""Combine all strategy summaries into one comparison table.

Run from the belief_state_trader folder:

    python scripts/08_strategy_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUMMARY_FILES = [
    "buy_hold_summary.csv",
    "moving_average_summary.csv",
    "single_regime_summary.csv",
    "bayesian_strategy_summary.csv",
]

MAIN_COLUMNS = [
    "strategy",
    "start_date",
    "end_date",
    "n_days",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "fixed_weight",
    "risk_aversion",
    "average_weight",
    "min_weight",
    "max_weight",
]

PERCENT_COLUMNS = [
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "fixed_weight",
    "average_weight",
    "min_weight",
    "max_weight",
]


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in PERCENT_COLUMNS:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
    if "sharpe" in display.columns:
        display["sharpe"] = display["sharpe"].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    if "risk_aversion" in display.columns:
        display["risk_aversion"] = display["risk_aversion"].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    return display


def main():
    result_dir = PROJECT_ROOT / "results"
    summaries = []

    for filename in SUMMARY_FILES:
        path = result_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the earlier scripts before comparison."
            )
        summaries.append(pd.read_csv(path))

    comparison = pd.concat(summaries, ignore_index=True, sort=False)
    ordered_columns = [column for column in MAIN_COLUMNS if column in comparison.columns]
    extra_columns = [column for column in comparison.columns if column not in ordered_columns]
    comparison = comparison[ordered_columns + extra_columns]

    output_path = result_dir / "strategy_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("Strategy comparison on test split")
    print("---------------------------------")
    print(format_for_display(comparison).to_string(index=False))
    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()
