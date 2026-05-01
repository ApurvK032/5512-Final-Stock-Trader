"""Plotting helpers for strategy and belief outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import backtest


def plot_equity_curves(strategy_returns: dict[str, pd.Series], output_path: Path) -> None:
    """Save one plot comparing strategy equity curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, log_returns in strategy_returns.items():
        clean_returns = log_returns.dropna()
        equity = backtest.equity_curve(clean_returns)
        ax.plot(equity.index, equity.values, label=name)

    ax.set_title("Strategy Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value, starting near 1.0")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_belief_probabilities(beliefs: pd.DataFrame, output_path: Path) -> None:
    """Save one plot showing daily hidden-state belief probabilities."""
    plot_data = beliefs.copy()
    if "Date" in plot_data.columns:
        plot_data["Date"] = pd.to_datetime(plot_data["Date"])
        plot_data = plot_data.set_index("Date")

    state_columns = [
        column
        for column in plot_data.columns
        if column.startswith("state_") and column.endswith("_prob")
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for column in state_columns:
        ax.plot(plot_data.index, plot_data[column], label=column)

    ax.set_title("Daily Regime Beliefs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
