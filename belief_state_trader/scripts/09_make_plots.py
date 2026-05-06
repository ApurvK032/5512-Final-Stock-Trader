"""Create simple plots for the end-to-end pipeline.

Run from the belief_state_trader folder:

    python scripts/09_make_plots.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import baselines, belief_update, data, plots, portfolio


RISK_AVERSION = 2.0
TRANSACTION_COST_RATE = 0.001


def bayesian_strategy_returns():
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
    weights = portfolio.weights_from_beliefs(beliefs, state_moments, risk_aversion=RISK_AVERSION)
    return portfolio.returns_from_weights(
        weights,
        test["Real_Log_Return"],
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )


def main():
    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    buy_hold_returns = baselines.buy_and_hold_returns(test["Real_Log_Return"])
    moving_average_returns = baselines.moving_average_crossover_returns(
        prices=test["Adj_Close"],
        real_log_returns=test["Real_Log_Return"],
        short_window=50,
        long_window=200,
    )
    single_regime_returns, _ = baselines.single_regime_no_belief_returns(
        train_real_log_returns=train["Real_Log_Return"],
        test_real_log_returns=test["Real_Log_Return"],
        risk_aversion=2.0,
    )

    strategy_returns = {
        "Buy and Hold": buy_hold_returns,
        "Moving Average 50/200": moving_average_returns,
        "Single-Regime No-Belief": single_regime_returns,
        "Bayesian Belief-State": bayesian_strategy_returns(),
    }

    result_dir = PROJECT_ROOT / "results"
    equity_plot_path = result_dir / "equity_curves.png"
    beliefs_plot_path = result_dir / "belief_probabilities.png"

    plots.plot_equity_curves(strategy_returns, equity_plot_path)

    beliefs_path = result_dir / "test_beliefs.csv"
    if not beliefs_path.exists():
        raise FileNotFoundError(
            f"Missing {beliefs_path}. Run scripts/06_bayesian_belief_update.py first."
        )
    beliefs = pd.read_csv(beliefs_path)
    plots.plot_belief_probabilities(beliefs, beliefs_plot_path)

    print("Generated pipeline plots")
    print("------------------------")
    print(f"Saved equity curves to: {equity_plot_path}")
    print(f"Saved belief probabilities to: {beliefs_plot_path}")


if __name__ == "__main__":
    main()
