"""Run all ablation studies.

Run from the belief_state_trader folder:

    python scripts/11_ablation_studies.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ablations, data


def main():
    results_dir = PROJECT_ROOT / "results"

    with open(results_dir / "hmm_model.pkl", "rb") as f:
        fitted = pickle.load(f)

    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    # ------------------------------------------------------------------
    # 1. Risk aversion sweep
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ABLATION 1: RISK AVERSION SWEEP")
    print("=" * 60)
    ra_results = ablations.risk_aversion_sweep(fitted, train, test)
    print(ra_results[
        ["strategy", "risk_aversion", "total_return", "sharpe", "sortino",
         "max_drawdown", "average_weight"]
    ].round(4).to_string(index=False))
    ra_results.to_csv(results_dir / "ablation_risk_aversion.csv", index=False)
    print(f"\nSaved to: {results_dir / 'ablation_risk_aversion.csv'}")

    # ------------------------------------------------------------------
    # 2. Feature ablations
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 2: FEATURE SUBSETS")
    print("=" * 60)
    feat_results = ablations.feature_ablation(train, test)
    print(feat_results[
        ["strategy", "features", "total_return", "sharpe", "sortino",
         "max_drawdown", "average_weight", "hmm_log_likelihood"]
    ].round(4).to_string(index=False))
    feat_results.to_csv(results_dir / "ablation_features.csv", index=False)
    print(f"\nSaved to: {results_dir / 'ablation_features.csv'}")

    # ------------------------------------------------------------------
    # 3. Entropy ablation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION 3: ENTROPY-AWARE POSITION SIZING")
    print("=" * 60)
    ent_results = ablations.entropy_ablation(fitted, train, test)
    print(ent_results[
        ["strategy", "method", "total_return", "sharpe", "sortino",
         "max_drawdown", "average_weight"]
    ].round(4).to_string(index=False))
    ent_results.to_csv(results_dir / "ablation_entropy.csv", index=False)
    print(f"\nSaved to: {results_dir / 'ablation_entropy.csv'}")

    print("\n" + "=" * 60)
    print("All ablation studies complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
