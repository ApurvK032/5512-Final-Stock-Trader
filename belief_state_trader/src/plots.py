"""Plotting helpers for strategy and belief outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src import backtest


CRISIS_COLOR = "#ffcccc"
STATE_COLORS = ["#d62728", "#ff7f0e", "#2ca02c"]
STATE_LABELS = ["Bear (State 0)", "Sideways (State 1)", "Bull (State 2)"]


def _shade_crisis_periods(ax, events: pd.DataFrame | None):
    """Add light red shading for manually labeled crisis periods."""
    if events is None:
        return
    for _, row in events.iterrows():
        ax.axvspan(row["start"], row["end"], alpha=0.15, color="red", zorder=0)


def plot_equity_curves(
    strategy_returns: dict[str, pd.Series],
    output_path: Path,
    events: pd.DataFrame | None = None,
) -> None:
    """Save one plot comparing strategy equity curves with optional crisis shading."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, log_returns in strategy_returns.items():
        clean_returns = log_returns.dropna()
        equity = backtest.equity_curve(clean_returns)
        ax.plot(equity.index, equity.values, label=name, linewidth=1.5)

    _shade_crisis_periods(ax, events)
    ax.set_title("Strategy Equity Curves (Test Period 2021–2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (starting at $1.00)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_belief_probabilities(
    beliefs: pd.DataFrame,
    output_path: Path,
    events: pd.DataFrame | None = None,
) -> None:
    """Save stacked area plot of daily regime beliefs."""
    plot_data = beliefs.copy()
    if "Date" in plot_data.columns:
        plot_data["Date"] = pd.to_datetime(plot_data["Date"])
        plot_data = plot_data.set_index("Date")

    state_columns = [
        c for c in plot_data.columns if c.startswith("state_") and c.endswith("_prob")
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        plot_data.index,
        *[plot_data[c].values for c in state_columns],
        labels=[STATE_LABELS[i] if i < len(STATE_LABELS) else c
                for i, c in enumerate(state_columns)],
        colors=STATE_COLORS[:len(state_columns)],
        alpha=0.8,
    )
    _shade_crisis_periods(ax, events)
    ax.set_title("Daily Regime Belief Distribution")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_entropy_and_weights(
    beliefs: pd.DataFrame,
    weights: pd.Series,
    output_path: Path,
    events: pd.DataFrame | None = None,
) -> None:
    """Two-panel plot: belief entropy (top) and portfolio weights (bottom)."""
    plot_data = beliefs.copy()
    if "Date" in plot_data.columns:
        plot_data["Date"] = pd.to_datetime(plot_data["Date"])
        plot_data = plot_data.set_index("Date")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: entropy
    ax1.plot(plot_data.index, plot_data["normalized_entropy"], color="#9467bd", linewidth=1)
    ax1.fill_between(plot_data.index, 0, plot_data["normalized_entropy"],
                      alpha=0.3, color="#9467bd")
    ax1.set_ylabel("Normalized Entropy")
    ax1.set_title("Belief Entropy (Uncertainty) and Portfolio Allocation Over Time")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    _shade_crisis_periods(ax1, events)

    # Bottom: weights
    ax2.plot(weights.index, weights.values, color="#1f77b4", linewidth=1)
    ax2.fill_between(weights.index, 0, weights.values, alpha=0.3, color="#1f77b4")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Weight")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    _shade_crisis_periods(ax2, events)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drawdown_comparison(
    strategy_returns: dict[str, pd.Series],
    output_path: Path,
    events: pd.DataFrame | None = None,
) -> None:
    """Plot drawdown curves for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, log_returns in strategy_returns.items():
        eq = backtest.equity_curve(log_returns.dropna())
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name, linewidth=1.2)

    _shade_crisis_periods(ax, events)
    ax.set_title("Strategy Drawdowns (Test Period 2021–2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_model_selection(model_selection_df: pd.DataFrame, output_path: Path) -> None:
    """BIC/AIC vs number of states with delta annotations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(model_selection_df["n_states"], model_selection_df["bic"],
             "o-", label="BIC", linewidth=2, markersize=8)
    ax1.plot(model_selection_df["n_states"], model_selection_df["aic"],
             "s--", label="AIC", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Hidden States (K)")
    ax1.set_ylabel("Information Criterion")
    ax1.set_title("Model Selection: BIC and AIC")
    ax1.set_xticks(model_selection_df["n_states"].tolist())
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    bic_deltas = model_selection_df["bic"].diff().iloc[1:]
    ax2.bar(model_selection_df["n_states"].iloc[1:], bic_deltas.abs(),
            color=["#2ca02c" if d < -500 else "#ff7f0e" if d < -200 else "#d62728"
                   for d in bic_deltas])
    ax2.set_xlabel("Number of Hidden States (K)")
    ax2.set_ylabel("|ΔBIC| from Previous K")
    ax2.set_title("Marginal BIC Improvement (Elbow Analysis)")
    ax2.set_xticks(model_selection_df["n_states"].iloc[1:].tolist())
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ablation_risk_aversion(ablation_df: pd.DataFrame, output_path: Path) -> None:
    """Risk aversion sweep: Sharpe/Sortino and drawdown vs lambda."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ablation_df["risk_aversion"], ablation_df["sharpe"],
             "o-", label="Sharpe", linewidth=2)
    ax1.plot(ablation_df["risk_aversion"], ablation_df["sortino"],
             "s--", label="Sortino", linewidth=2)
    ax1.set_xlabel("Risk Aversion (λ)")
    ax1.set_ylabel("Ratio")
    ax1.set_title("Risk-Adjusted Returns vs Risk Aversion")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ablation_df["risk_aversion"], ablation_df["max_drawdown"],
             "o-", color="#d62728", linewidth=2)
    ax2.set_xlabel("Risk Aversion (λ)")
    ax2.set_ylabel("Max Drawdown")
    ax2.set_title("Maximum Drawdown vs Risk Aversion")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration(calibration_df: pd.DataFrame, output_path: Path) -> None:
    """Plot target vs actual coverage for belief calibration."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(calibration_df["target_coverage"], calibration_df["actual_coverage"],
            "o-", linewidth=2, markersize=10, color="#1f77b4",
            label="Bayesian agent")
    for _, row in calibration_df.iterrows():
        ax.annotate(
            f"{row['actual_coverage']:.1%}",
            (row["target_coverage"], row["actual_coverage"]),
            textcoords="offset points", xytext=(10, -5), fontsize=10,
        )
    ax.set_xlabel("Target Coverage")
    ax.set_ylabel("Actual Coverage")
    ax.set_title("Belief Calibration: Predicted vs Actual Coverage")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_ablation(ablation_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart comparing feature subset performance."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(ablation_df))
    width = 0.25
    ax.bar([i - width for i in x], ablation_df["sharpe"], width, label="Sharpe")
    ax.bar(x, ablation_df["sortino"], width, label="Sortino")
    ax.bar([i + width for i in x], ablation_df["max_drawdown"].abs(), width,
           label="|Max Drawdown|", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ablation_df["strategy"].tolist(), rotation=15)
    ax.set_ylabel("Value")
    ax.set_title("Feature Ablation: Performance by Feature Subset")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
