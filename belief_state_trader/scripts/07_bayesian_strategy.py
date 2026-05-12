"""Backtest the baseline Bayesian belief-state strategy."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import backtest, belief_update, data, portfolio


def main():
    with open(PROJECT_ROOT / "results" / "hmm_model.pkl", "rb") as f:
        fitted = pickle.load(f)

    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    train_features = data.feature_matrix(train, fitted.feature_columns)
    valid_train = train["Real_Log_Return"].notna()
    train_states = fitted.model.predict(train_features.loc[valid_train].to_numpy(dtype=float))
    state_moments = portfolio.estimate_state_return_moments(
        states=train_states,
        real_log_returns=train.loc[valid_train, "Real_Log_Return"],
        n_states=fitted.model.n_components,
    )

    test_features = data.feature_matrix(test, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(test_features, fitted.model)
    weights = portfolio.weights_from_beliefs(beliefs, state_moments, risk_aversion=2.0)
    strategy_returns = portfolio.returns_from_weights(weights, test["Real_Log_Return"])

    summary = backtest.summarize_backtest("Bayesian Belief-State", strategy_returns)
    summary["risk_aversion"] = 2.0
    summary["average_weight"] = float(weights.mean())
    summary["min_weight"] = float(weights.min())
    summary["max_weight"] = float(weights.max())

    summary_df = pd.DataFrame([summary])
    summary_path = PROJECT_ROOT / "results" / "bayesian_strategy_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    daily_path = PROJECT_ROOT / "results" / "bayesian_strategy_daily.csv"
    pd.DataFrame(
        {
            "Date": weights.index,
            "weight": weights.to_numpy(),
            "strategy_log_return": (
                weights.shift(1).fillna(0.0) * test["Real_Log_Return"].reindex(weights.index)
            ).to_numpy(),
        }
    ).to_csv(daily_path, index=False)

    print("Bayesian belief-state strategy on test split")
    print("--------------------------------------------")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved daily weights/returns to: {daily_path}")


if __name__ == "__main__":
    main()

