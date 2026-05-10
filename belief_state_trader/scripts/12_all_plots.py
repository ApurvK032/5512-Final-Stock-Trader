"""Generate the consolidated figure set used in the final report workflow.

This script reconstructs baseline and Bayesian strategy return series, loads event
annotations, and produces publication-ready plots from previously generated
artifacts in results/. It is designed as a post-processing visualization step
after model fitting, belief updates, strategy evaluation, and (optionally)
ablation/model-selection analyses.

Required inputs:
- results/hmm_model.pkl
- results/test_beliefs.csv
- train/test CSVs loaded via src.data helpers

Always generated plots:
- results/equity_curves.png
- results/belief_probabilities.png
- results/entropy_and_weights.png
- results/drawdown_comparison.png

Conditionally generated plots (only if source CSV exists):
- results/model_selection_detailed.png from results/model_selection.csv
- results/ablation_risk_aversion.png from results/ablation_risk_aversion.csv
- results/calibration_plot.png from results/belief_calibration.csv
- results/ablation_features.png from results/ablation_features.csv

Run from the belief_state_trader folder:

    python scripts/12_all_plots.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import baselines, belief_update, data, evaluation, plots, portfolio


def main():
    results_dir = PROJECT_ROOT / "results"

    with open(results_dir / "hmm_model.pkl", "rb") as f:
        fitted = pickle.load(f)

    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())
    events = evaluation.get_manual_events()

    # --- Rebuild strategy returns ---
    buy_hold = baselines.buy_and_hold_returns(test["Real_Log_Return"])
    ma = baselines.moving_average_crossover_returns(
        test["Adj_Close"], test["Real_Log_Return"],
    )
    single, _ = baselines.single_regime_no_belief_returns(
        train["Real_Log_Return"], test["Real_Log_Return"],
    )

    train_features = data.feature_matrix(train, fitted.feature_columns)
    valid = train["Real_Log_Return"].notna()
    train_states = fitted.model.predict(train_features.loc[valid].to_numpy(dtype=float))
    state_moments = portfolio.estimate_state_return_moments(
        train_states, train.loc[valid, "Real_Log_Return"], fitted.model.n_components,
    )

    test_features = data.feature_matrix(test, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(test_features, fitted.model)
    weights = portfolio.weights_from_beliefs(beliefs, state_moments, risk_aversion=2.0)
    bayesian = portfolio.returns_from_weights(weights, test["Real_Log_Return"])

    strategy_returns = {
        "Buy and Hold": buy_hold,
        "Moving Average 50/200": ma,
        "Single-Regime No-Belief": single,
        "Bayesian Belief-State": bayesian,
    }

    # --- Generate all plots ---
    print("Generating plots...")

    plots.plot_equity_curves(strategy_returns, results_dir / "equity_curves.png", events)
    print("  equity_curves.png")

    beliefs_csv = pd.read_csv(results_dir / "test_beliefs.csv")
    plots.plot_belief_probabilities(beliefs_csv, results_dir / "belief_probabilities.png", events)
    print("  belief_probabilities.png")

    plots.plot_entropy_and_weights(beliefs, weights, results_dir / "entropy_and_weights.png", events)
    print("  entropy_and_weights.png")

    plots.plot_drawdown_comparison(strategy_returns, results_dir / "drawdown_comparison.png", events)
    print("  drawdown_comparison.png")

    ms_path = results_dir / "model_selection.csv"
    if ms_path.exists():
        ms_df = pd.read_csv(ms_path)
        plots.plot_model_selection(ms_df, results_dir / "model_selection_detailed.png")
        print("  model_selection_detailed.png")

    ra_path = results_dir / "ablation_risk_aversion.csv"
    if ra_path.exists():
        ra_df = pd.read_csv(ra_path)
        plots.plot_ablation_risk_aversion(ra_df, results_dir / "ablation_risk_aversion.png")
        print("  ablation_risk_aversion.png")

    cal_path = results_dir / "belief_calibration.csv"
    if cal_path.exists():
        cal_df = pd.read_csv(cal_path)
        plots.plot_calibration(cal_df, results_dir / "calibration_plot.png")
        print("  calibration_plot.png")

    feat_path = results_dir / "ablation_features.csv"
    if feat_path.exists():
        feat_df = pd.read_csv(feat_path)
        plots.plot_feature_ablation(feat_df, results_dir / "ablation_features.png")
        print("  ablation_features.png")

    print(f"\nAll plots saved to {results_dir}/")


if __name__ == "__main__":
    main()
