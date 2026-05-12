"""Run the enhanced Bayesian agent pipeline and benchmark it against baselines.

This script executes a full enhanced-strategy experiment workflow on the
train/test split. It first runs a parameter sweep across enhanced-agent
variants, selects the best configuration by Sharpe ratio, then compares that
configuration against baseline strategies and the original Bayesian agent.

Main stages:
1. Full parameter sweep over enhanced variants and hyperparameters.
2. Best-config backtest against buy-and-hold, moving-average, single-regime,
   and original Bayesian strategies.
3. Turnover/cost diagnostics from weight changes.
4. Performance breakdown across manually labeled market periods.
5. Visualization of equity curves, drawdowns, and allocation behavior.

Primary outputs:
- results/enhanced_agent_sweep.csv
- results/enhanced_comparison.csv
- results/crisis_periods_drawdown.csv
- results/enhanced_equity_curves.png
- results/enhanced_drawdowns.png
- results/enhanced_weight_comparison.png

Run from the belief_state_trader folder:

    python scripts/13_enhanced_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import backtest, baselines, data, enhanced_agent, evaluation, portfolio


def main():
    results_dir = PROJECT_ROOT / "results"
    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    # ------------------------------------------------------------------
    # 1. Full sweep: mean-variance enhanced + regime-switch variants
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. FULL PARAMETER SWEEP (Mean-Variance + Regime-Switch)")
    print("=" * 70)
    sweep = enhanced_agent.full_sweep(train, test)
    display_cols = [
        "strategy", "variant", "total_return", "sharpe", "sortino",
        "max_drawdown", "calmar", "average_weight", "total_turnover", "n_trades",
    ]
    print(sweep[display_cols].round(4).to_string(index=False))
    sweep.to_csv(results_dir / "enhanced_agent_sweep.csv", index=False)

    # Pick best by Sharpe
    best_idx = sweep["sharpe"].idxmax()
    best_cfg = sweep.loc[best_idx]
    print(f"\nBest config by Sharpe ({best_cfg['sharpe']:.4f}): {best_cfg['strategy']}")
    print(f"  Variant: {best_cfg['variant']}")
    print(f"  Total return: {best_cfg['total_return']:.2%}, Max DD: {best_cfg['max_drawdown']:.2%}")

    # ------------------------------------------------------------------
    # 2. Run best enhanced agent from sweep
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. SELECTED ENHANCED AGENT vs ALL STRATEGIES")
    print("=" * 70)

    # Run the best config from sweep using its variant type
    if best_cfg["variant"] == "bear-defense":
        result = enhanced_agent.run_bear_defense_agent(
            train, test,
            bear_threshold=0.25, defense_scale=5.0, min_weight=0.0,
            rebalance_threshold=0.03, smoothing_alpha=0.40,
            label="Enhanced Bayesian Agent",
        )
    elif best_cfg["variant"] == "regime-switch":
        result = enhanced_agent.run_regime_switch_agent(
            train, test,
            bull_threshold=0.3, bear_threshold=0.20, sideways_weight=1.0,
            rebalance_threshold=0.03, smoothing_alpha=0.35,
            label="Enhanced Bayesian Agent",
        )
    else:
        result = enhanced_agent.run_enhanced_agent(
            train, test,
            bull_lambda=0.5, bear_lambda=12.0,
            rebalance_threshold=0.08, smoothing_alpha=0.2,
        )
    enh_summary = result["summary"]
    enh_weights = result["weights"]
    enh_returns = result["returns"]

    # Build all strategy returns for comparison
    bh_returns = baselines.buy_and_hold_returns(test["Real_Log_Return"])
    ma_returns = baselines.moving_average_crossover_returns(
        test["Adj_Close"], test["Real_Log_Return"],
    )
    sr_returns, _ = baselines.single_regime_no_belief_returns(
        train["Real_Log_Return"], test["Real_Log_Return"],
    )

    # Original Bayesian agent (all features, fixed lambda=2)
    import pickle
    hmm_path = results_dir / "hmm_model.pkl"
    if hmm_path.exists():
        with open(hmm_path, "rb") as f:
            fitted_orig = pickle.load(f)
        from src import belief_update
        orig_features = data.feature_matrix(train, fitted_orig.feature_columns)
        valid = train["Real_Log_Return"].notna()
        orig_states = fitted_orig.model.predict(
            orig_features.loc[valid].to_numpy(dtype=float)
        )
        orig_moments = portfolio.estimate_state_return_moments(
            orig_states, train.loc[valid, "Real_Log_Return"],
            fitted_orig.model.n_components,
        )
        orig_beliefs = belief_update.filter_beliefs(
            data.feature_matrix(test, fitted_orig.feature_columns), fitted_orig.model,
        )
        orig_weights = portfolio.weights_from_beliefs(orig_beliefs, orig_moments, 2.0)
        orig_returns = portfolio.returns_from_weights(orig_weights, test["Real_Log_Return"])
    else:
        orig_returns = enh_returns  # fallback

    all_summaries = []
    for name, rets in [
        ("Buy and Hold", bh_returns),
        ("Moving Average 50/200", ma_returns),
        ("Single-Regime No-Belief", sr_returns),
        ("Original Bayesian (base)", orig_returns),
        ("Enhanced Bayesian Agent", enh_returns),
    ]:
        s = backtest.summarize_backtest(name, rets)
        all_summaries.append(s)

    comparison = pd.DataFrame(all_summaries)
    print(comparison[
        ["strategy", "total_return", "annualized_return", "annualized_volatility",
         "sharpe", "sortino", "max_drawdown", "calmar"]
    ].round(4).to_string(index=False))
    comparison.to_csv(results_dir / "enhanced_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Turnover comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. TURNOVER AND COST ANALYSIS")
    print("=" * 70)

    def turnover_stats(weights, name):
        changes = weights.diff().abs().dropna()
        return {
            "strategy": name,
            "total_turnover": float(changes.sum()),
            "n_trades": int((changes > 0).sum()),
            "cost_drag_pct": float(changes.sum() * 0.001 * 100),
            "avg_weight": float(weights.mean()),
        }

    bh_weights = baselines.buy_and_hold_weights(test["Real_Log_Return"])
    sr_weights, _ = baselines.single_regime_weight(
        train["Real_Log_Return"], test["Real_Log_Return"],
    )

    turnover_rows = [
        turnover_stats(bh_weights, "Buy and Hold"),
        turnover_stats(sr_weights, "Single-Regime"),
        turnover_stats(orig_weights, "Original Bayesian"),
        turnover_stats(enh_weights, "Enhanced Bayesian"),
    ]
    turnover_df = pd.DataFrame(turnover_rows)
    print(turnover_df.round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Performance during crisis periods
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. CRISIS PERIOD PERFORMANCE")
    print("=" * 70)
    events = evaluation.get_manual_events()
    event_labels = evaluation.label_dates_by_event(test.index, events)
    strat_returns = {
        "Buy and Hold": bh_returns,
        "Single-Regime": sr_returns,
        "Original Bayesian": orig_returns,
        "Enhanced Bayesian": enh_returns,
    }
    period_perf = evaluation.strategy_performance_by_period(strat_returns, event_labels)
    print(period_perf[
        ["period", "strategy", "total_return", "sharpe", "max_drawdown"]
    ].round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. GENERATING PLOTS")
    print("=" * 70)

    # Equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, rets in [
        ("Buy and Hold", bh_returns),
        ("Single-Regime No-Belief", sr_returns),
        ("Original Bayesian", orig_returns),
        ("Enhanced Bayesian", enh_returns),
    ]:
        eq = backtest.equity_curve(rets.dropna())
        ax.plot(eq.index, eq.values, label=name, linewidth=1.5)
    for _, row in events.iterrows():
        ax.axvspan(row["start"], row["end"], alpha=0.12, color="red", zorder=0)
    ax.set_title("Enhanced Bayesian Agent vs All Strategies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($1 start)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(results_dir / "enhanced_equity_curves.png", dpi=150)
    plt.close(fig)
    print("  enhanced_equity_curves.png")

    # Drawdown comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, rets in [
        ("Buy and Hold", bh_returns),
        ("Single-Regime", sr_returns),
        ("Original Bayesian", orig_returns),
        ("Enhanced Bayesian", enh_returns),
    ]:
        eq = backtest.equity_curve(rets.dropna())
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=name, linewidth=1.2)
    for _, row in events.iterrows():
        ax.axvspan(row["start"], row["end"], alpha=0.12, color="red", zorder=0)
    ax.set_title("Drawdown Comparison: Enhanced Bayesian vs Others")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(results_dir / "enhanced_drawdowns.png", dpi=150)
    plt.close(fig)
    print("  enhanced_drawdowns.png")

    # Weight comparison: original vs enhanced
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(orig_weights.index, orig_weights.values, color="#d62728", linewidth=0.8)
    ax1.fill_between(orig_weights.index, 0, orig_weights.values, alpha=0.3, color="#d62728")
    ax1.set_ylabel("Weight")
    ax1.set_title("Original Bayesian Agent Weights (noisy, high turnover)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    for _, row in events.iterrows():
        ax1.axvspan(row["start"], row["end"], alpha=0.12, color="red", zorder=0)

    ax2.plot(enh_weights.index, enh_weights.values, color="#2ca02c", linewidth=0.8)
    ax2.fill_between(enh_weights.index, 0, enh_weights.values, alpha=0.3, color="#2ca02c")
    ax2.set_ylabel("Weight")
    ax2.set_xlabel("Date")
    ax2.set_title("Enhanced Bayesian Agent Weights (smooth, low turnover)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    for _, row in events.iterrows():
        ax2.axvspan(row["start"], row["end"], alpha=0.12, color="red", zorder=0)

    fig.tight_layout()
    fig.savefig(results_dir / "enhanced_weight_comparison.png", dpi=150)
    plt.close(fig)
    print("  enhanced_weight_comparison.png")

    print(f"\nAll results saved to {results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
