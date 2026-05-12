"""Compare the HMM belief filter with a pgmpy DBN version."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import backtest, belief_update, data, portfolio

try:
    from src import dbn_model
    DBN_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name == "pgmpy" or exc.name.startswith("pgmpy."):
        DBN_AVAILABLE = False
    else:
        raise


def main():
    if not DBN_AVAILABLE:
        print("pgmpy is not installed; skipping 11_pgmpy_dbn.py.")
        print("Install with: pip install pgmpy")
        return

    results_dir = PROJECT_ROOT / "results"

    with open(results_dir / "hmm_model.pkl", "rb") as f:
        fitted = pickle.load(f)

    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    # ------------------------------------------------------------------
    # 1. Build DBN from fitted HMM
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. DBN STRUCTURE (from HMM parameters)")
    print("=" * 60)
    dbn = dbn_model.build_dbn_from_hmm(fitted)
    summary = dbn_model.dbn_structure_summary(fitted)

    print(f"Hidden states: {summary['n_states']}")
    print(f"Features: {summary['feature_names']}")
    print(f"Intra-slice edges: {summary['intra_slice_edges']}")
    print(f"Inter-slice edges: {summary['inter_slice_edges']}")
    print(f"\nStart probs: {[round(p, 4) for p in summary['start_probs']]}")
    print("\nState parameters:")
    for sp in summary["state_parameters"]:
        print(f"  State {sp['state']}: mean={[round(m,4) for m in sp['mean']]}, "
              f"var_diag={[round(v,4) for v in sp['covariance_diag']]}")

    with open(results_dir / "dbn_structure.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # 2. Run DBN forward beliefs on test data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. DBN FORWARD BELIEFS (test period)")
    print("=" * 60)
    test_features = data.feature_matrix(test, fitted.feature_columns)

    dbn_beliefs = dbn_model.dbn_forward_beliefs(fitted, test_features)
    dbn_beliefs_out = dbn_beliefs.reset_index()
    dbn_beliefs_out.to_csv(results_dir / "dbn_beliefs.csv", index=False)
    print(dbn_beliefs_out.head().to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Compare HMM beliefs vs DBN beliefs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. HMM vs DBN BELIEF COMPARISON")
    print("=" * 60)
    hmm_beliefs = belief_update.filter_beliefs(test_features, fitted.model)
    comparison = dbn_model.compare_belief_methods(hmm_beliefs, dbn_beliefs)
    for k, v in comparison.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    pd.DataFrame([comparison]).to_csv(results_dir / "hmm_vs_dbn_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Backtest the DBN-based strategy
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. DBN-BASED STRATEGY BACKTEST")
    print("=" * 60)
    train_features = data.feature_matrix(train, fitted.feature_columns)
    valid = train["Real_Log_Return"].notna()
    train_states = fitted.model.predict(train_features.loc[valid].to_numpy(dtype=float))
    state_moments = portfolio.estimate_state_return_moments(
        train_states, train.loc[valid, "Real_Log_Return"], fitted.model.n_components,
    )

    dbn_weights = portfolio.weights_from_beliefs(dbn_beliefs, state_moments, risk_aversion=2.0)
    dbn_returns = portfolio.returns_from_weights(dbn_weights, test["Real_Log_Return"])
    dbn_summary = backtest.summarize_backtest("DBN Belief-State", dbn_returns)
    dbn_summary["average_weight"] = float(dbn_weights.mean())

    hmm_weights = portfolio.weights_from_beliefs(hmm_beliefs, state_moments, risk_aversion=2.0)
    hmm_returns = portfolio.returns_from_weights(hmm_weights, test["Real_Log_Return"])
    hmm_summary = backtest.summarize_backtest("HMM Belief-State", hmm_returns)
    hmm_summary["average_weight"] = float(hmm_weights.mean())

    combined = pd.DataFrame([hmm_summary, dbn_summary])
    print(combined[["strategy", "total_return", "sharpe", "sortino",
                     "max_drawdown", "average_weight"]].round(4).to_string(index=False))
    combined.to_csv(results_dir / "hmm_vs_dbn_backtest.csv", index=False)

    print("\n" + "=" * 60)
    print("All pgmpy DBN results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
