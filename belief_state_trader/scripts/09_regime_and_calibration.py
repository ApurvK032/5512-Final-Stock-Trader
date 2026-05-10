"""Evaluate regime quality, belief calibration, and period-specific performance.

This script is an evaluation layer on top of the fitted HMM and Bayesian belief
pipeline. It measures whether inferred regimes align with stress periods,
whether belief probabilities are calibrated to realized returns, and how
different strategies behave across market conditions.

Main analyses:
1. Drawdown-based crisis detection from test-period prices.
2. Manual event labeling and date-to-period assignment.
3. Regime detection accuracy for the inferred bear state.
4. Belief calibration diagnostics against realized returns.
5. Strategy performance by labeled period (transition-aware comparison).
6. Belief entropy behavior across periods (uncertainty analysis).

Required inputs:
- results/hmm_model.pkl
- train/test data loaded via src.data

Generated outputs:
- results/crisis_periods_drawdown.csv
- results/regime_detection_accuracy.csv
- results/belief_calibration.csv
- results/performance_by_period.csv
- results/entropy_by_period.csv

Run from the belief_state_trader folder:

    python scripts/09_regime_and_calibration.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    backtest,
    baselines,
    belief_update,
    data,
    evaluation,
    portfolio,
)


def build_strategy_returns(fitted, train, test, state_moments):
    """Reconstruct all strategy return series for period analysis."""
    buy_hold = baselines.buy_and_hold_returns(test["Real_Log_Return"])
    ma = baselines.moving_average_crossover_returns(
        prices=test["Adj_Close"],
        real_log_returns=test["Real_Log_Return"],
    )
    single, _ = baselines.single_regime_no_belief_returns(
        train["Real_Log_Return"], test["Real_Log_Return"],
    )

    test_features = data.feature_matrix(test, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(test_features, fitted.model)
    weights = portfolio.weights_from_beliefs(beliefs, state_moments, risk_aversion=2.0)
    bayesian = portfolio.returns_from_weights(weights, test["Real_Log_Return"])

    return {
        "Buy and Hold": buy_hold,
        "Moving Average 50/200": ma,
        "Single-Regime No-Belief": single,
        "Bayesian Belief-State": bayesian,
    }


def identify_bear_state(fitted, train):
    """Determine which HMM state has the lowest mean return (bear)."""
    train_features = data.feature_matrix(train, fitted.feature_columns)
    valid = train["Real_Log_Return"].notna()
    states = fitted.model.predict(train_features.loc[valid].to_numpy(dtype=float))
    moments = portfolio.estimate_state_return_moments(
        states, train.loc[valid, "Real_Log_Return"], fitted.model.n_components,
    )
    return int(moments["mean_return"].idxmin())


def main():
    results_dir = PROJECT_ROOT / "results"

    with open(results_dir / "hmm_model.pkl", "rb") as f:
        fitted = pickle.load(f)

    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    # --- Beliefs and state moments ---
    train_features = data.feature_matrix(train, fitted.feature_columns)
    valid_train = train["Real_Log_Return"].notna()
    train_states = fitted.model.predict(
        train_features.loc[valid_train].to_numpy(dtype=float)
    )
    state_moments = portfolio.estimate_state_return_moments(
        train_states, train.loc[valid_train, "Real_Log_Return"],
        fitted.model.n_components,
    )

    test_features = data.feature_matrix(test, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(test_features, fitted.model)
    bear_state = identify_bear_state(fitted, train)

    # ------------------------------------------------------------------
    # 1. Drawdown-based crisis detection
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. DRAWDOWN-BASED CRISIS DETECTION")
    print("=" * 60)
    crisis_periods = evaluation.detect_crisis_periods(test["Adj_Close"], threshold=-0.10)
    print(crisis_periods.to_string(index=False))
    crisis_periods.to_csv(results_dir / "crisis_periods_drawdown.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Manual event labels
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. MANUAL EVENT LABELS")
    print("=" * 60)
    events = evaluation.get_manual_events()
    print(events[["name", "start", "end", "type"]].to_string(index=False))
    event_labels = evaluation.label_dates_by_event(test.index, events)
    event_counts = event_labels.value_counts()
    print(f"\nDays per label:\n{event_counts.to_string()}")

    # ------------------------------------------------------------------
    # 3. Regime detection accuracy
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"3. REGIME DETECTION ACCURACY (bear_state={bear_state})")
    print("=" * 60)
    accuracy = evaluation.regime_detection_accuracy(beliefs, event_labels, bear_state)
    for k, v in accuracy.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    pd.DataFrame([accuracy]).to_csv(results_dir / "regime_detection_accuracy.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Belief calibration
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. BELIEF CALIBRATION")
    print("=" * 60)
    cal = evaluation.belief_calibration(
        beliefs, state_moments, test["Real_Log_Return"],
    )
    print(cal.to_string(index=False))
    cal.to_csv(results_dir / "belief_calibration.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Strategy performance during regime transitions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. STRATEGY PERFORMANCE BY PERIOD")
    print("=" * 60)
    strat_returns = build_strategy_returns(fitted, train, test, state_moments)
    period_perf = evaluation.strategy_performance_by_period(strat_returns, event_labels)
    print(period_perf.round(4).to_string(index=False))
    period_perf.to_csv(results_dir / "performance_by_period.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Entropy during transitions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. BELIEF ENTROPY DURING TRANSITIONS")
    print("=" * 60)
    entropy_analysis = evaluation.entropy_during_transitions(beliefs, event_labels)
    print(entropy_analysis.round(4).to_string(index=False))
    entropy_analysis.to_csv(results_dir / "entropy_by_period.csv", index=False)

    print("\n" + "=" * 60)
    print("All evaluation results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
