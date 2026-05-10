"""Validate the enhanced Bayesian strategy on a held-out period.

This script keeps Nicole's enhanced strategy logic, but separates parameter
selection from final evaluation:

1. Fit the enhanced HMM on 2015-2020 training data only.
2. Select enhanced strategy parameters on 2021-2022 validation data.
3. Report the selected strategy on 2023-2025 held-out data.

Run from the belief_state_trader folder:

    python scripts/14_validate_enhanced_strategy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import backtest, baselines, belief_update, data, enhanced_agent, portfolio


TRANSACTION_COST_BPS = 10.0
VALIDATION_START = "2021-01-05"
VALIDATION_END = "2022-12-30"
HELDOUT_START = "2023-01-03"
HELDOUT_END = "2025-12-30"
FULL_TEST_START = "2021-01-05"
FULL_TEST_END = "2025-12-30"


def date_window(series: pd.Series, start_date: str, end_date: str) -> pd.Series:
    """Return one date window from a date-indexed series."""
    return series.loc[
        (series.index >= pd.Timestamp(start_date))
        & (series.index <= pd.Timestamp(end_date))
    ]


def enhanced_configs() -> list[dict]:
    """Small grid based on the strongest enhanced strategy ideas."""
    configs: list[dict] = []

    for bull_threshold in [0.3, 0.5]:
        for bear_threshold in [0.10, 0.15, 0.20]:
            for sideways_weight in [0.80, 0.95, 1.00]:
                configs.append(
                    {
                        "strategy": (
                            f"RS bull>{bull_threshold}, bear>{bear_threshold}, "
                            f"sideways={sideways_weight}"
                        ),
                        "variant": "regime-switch",
                        "bull_threshold": bull_threshold,
                        "bear_threshold": bear_threshold,
                        "sideways_weight": sideways_weight,
                        "rebalance_threshold": 0.03,
                        "smoothing_alpha": 0.35,
                    }
                )

    for bear_threshold in [0.10, 0.15, 0.20, 0.25]:
        for defense_scale in [3.0, 4.0, 5.0]:
            configs.append(
                {
                    "strategy": (
                        f"BD bear>{bear_threshold}, scale={defense_scale}, "
                        "min=0.0"
                    ),
                    "variant": "bear-defense",
                    "bear_threshold": bear_threshold,
                    "defense_scale": defense_scale,
                    "min_weight": 0.0,
                    "rebalance_threshold": 0.03,
                    "smoothing_alpha": 0.35,
                }
            )

    return configs


def weights_for_config(
    config: dict,
    beliefs: pd.DataFrame,
    state_roles: dict,
) -> pd.Series:
    """Build enhanced strategy weights for one config."""
    if config["variant"] == "bear-defense":
        return enhanced_agent.bear_defense_weights(
            beliefs,
            state_roles,
            bear_threshold=config["bear_threshold"],
            defense_scale=config["defense_scale"],
            min_weight=config["min_weight"],
            rebalance_threshold=config["rebalance_threshold"],
            smoothing_alpha=config["smoothing_alpha"],
        )

    if config["variant"] == "regime-switch":
        return enhanced_agent.regime_switch_weights(
            beliefs,
            state_roles,
            bull_threshold=config["bull_threshold"],
            bear_threshold=config["bear_threshold"],
            sideways_weight=config["sideways_weight"],
            rebalance_threshold=config["rebalance_threshold"],
            smoothing_alpha=config["smoothing_alpha"],
        )

    raise ValueError(f"Unknown strategy variant: {config['variant']}")


def summarize_window(
    strategy_name: str,
    returns: pd.Series,
    weights: pd.Series,
    window_name: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Summarize returns, exposure, turnover, and costs for one window."""
    window_returns = date_window(returns, start_date, end_date).dropna()
    window_weights = weights.reindex(window_returns.index)
    window_turnover = weights.diff().abs().reindex(window_returns.index).fillna(0.0)

    summary = backtest.summarize_backtest(strategy_name, window_returns)
    summary["evaluation_window"] = window_name
    summary["average_weight"] = float(window_weights.mean())
    summary["average_turnover"] = float(window_turnover.mean())
    summary["total_turnover"] = float(window_turnover.sum())
    summary["total_transaction_cost"] = float(
        window_turnover.sum() * TRANSACTION_COST_BPS / 10_000.0
    )
    return summary


def add_config_columns(summary: dict, config: dict) -> dict:
    """Attach config values to a summary row."""
    row = dict(summary)
    for key, value in config.items():
        if key != "strategy":
            row[key] = value
    return row


def evaluate_configs(
    configs: list[dict],
    beliefs: pd.DataFrame,
    state_roles: dict,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all enhanced configs on validation, held-out, and full test windows."""
    rows = []
    windows = [
        ("validation_2021_2022", VALIDATION_START, VALIDATION_END),
        ("heldout_2023_2025", HELDOUT_START, HELDOUT_END),
        ("full_test_2021_2025", FULL_TEST_START, FULL_TEST_END),
    ]

    for config in configs:
        weights = weights_for_config(config, beliefs, state_roles)
        returns = portfolio.returns_from_weights(
            weights,
            test_df["Real_Log_Return"],
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
        for window_name, start_date, end_date in windows:
            summary = summarize_window(
                strategy_name=config["strategy"],
                returns=returns,
                weights=weights,
                window_name=window_name,
                start_date=start_date,
                end_date=end_date,
            )
            rows.append(add_config_columns(summary, config))

    return pd.DataFrame(rows)


def baseline_weights(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return baseline weight series for fair held-out comparison."""
    buy_weights = baselines.buy_and_hold_weights(test_df["Real_Log_Return"])
    ma_weights = baselines.moving_average_weights(test_df["Adj_Close"])
    single_weights, _ = baselines.single_regime_weight(
        train_df["Real_Log_Return"],
        test_df["Real_Log_Return"],
    )
    return {
        "Buy and Hold": buy_weights,
        "Moving Average 50/200": ma_weights,
        "Single-Regime No-Belief": single_weights,
    }


def main() -> None:
    result_dir = PROJECT_ROOT / "results"
    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    fitted = enhanced_agent.fit_enhanced_hmm(train)
    state_roles = enhanced_agent.identify_state_roles(fitted, train)
    beliefs = belief_update.filter_beliefs(
        data.feature_matrix(test, fitted.feature_columns),
        fitted.model,
    )

    configs = enhanced_configs()
    sweep = evaluate_configs(configs, beliefs, state_roles, test)
    sweep_path = result_dir / "enhanced_validation_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    validation = sweep[sweep["evaluation_window"] == "validation_2021_2022"]
    selected = validation.sort_values(
        ["sharpe", "max_drawdown", "total_return"],
        ascending=[False, False, False],
    ).iloc[0]

    selected_rows = sweep[
        (sweep["strategy"] == selected["strategy"])
        & (sweep["variant"] == selected["variant"])
    ].copy()
    selected_path = result_dir / "enhanced_validation_selected.csv"
    selected_rows.to_csv(selected_path, index=False)

    comparison_rows = []
    for name, weights in baseline_weights(train, test).items():
        returns = portfolio.returns_from_weights(
            weights,
            test["Real_Log_Return"],
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
        comparison_rows.append(
            summarize_window(
                strategy_name=name,
                returns=returns,
                weights=weights,
                window_name="heldout_2023_2025",
                start_date=HELDOUT_START,
                end_date=HELDOUT_END,
            )
        )

    selected_heldout = selected_rows[
        selected_rows["evaluation_window"] == "heldout_2023_2025"
    ].iloc[0]
    comparison_rows.append(selected_heldout.to_dict())

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = result_dir / "enhanced_validation_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    full_best = sweep[sweep["evaluation_window"] == "full_test_2021_2025"].sort_values(
        ["sharpe", "max_drawdown", "total_return"],
        ascending=[False, False, False],
    ).iloc[0]

    display_cols = [
        "strategy",
        "evaluation_window",
        "total_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "average_weight",
        "average_turnover",
        "total_transaction_cost",
    ]

    print("Enhanced strategy validation check")
    print("----------------------------------")
    print(f"feature columns: {fitted.feature_columns}")
    print(f"state roles: {state_roles}")
    print(f"tested enhanced configs: {len(configs)}")
    print("\nSelected using validation only:")
    print(selected[display_cols].to_string())
    print("\nSame selected strategy on held-out period:")
    print(selected_heldout[display_cols].to_string())
    print("\nBest strategy if selected on the full 2021-2025 test period:")
    print(full_best[display_cols].to_string())
    print("\nHeld-out comparison:")
    print(comparison[display_cols].round(4).to_string(index=False))
    print(f"\nSaved sweep to: {sweep_path}")
    print(f"Saved selected rows to: {selected_path}")
    print(f"Saved held-out comparison to: {comparison_path}")


if __name__ == "__main__":
    main()
