"""Run Bayesian belief updates on the test split.

Run from the belief_state_trader folder:

    python scripts/06_bayesian_belief_update.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import belief_update, data


def main():
    with open(PROJECT_ROOT / "results" / "hmm_model.pkl", "rb") as f:
        fitted = pickle.load(f)

    test = data.load_test()
    features = data.feature_matrix(test, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(features, fitted.model)

    output = beliefs.reset_index()
    output_path = PROJECT_ROOT / "results" / "test_beliefs.csv"
    output.to_csv(output_path, index=False)

    print("Bayesian belief updates on test split")
    print("-------------------------------------")
    print(output.head().to_string(index=False))
    print(f"\nSaved beliefs to: {output_path}")


if __name__ == "__main__":
    main()

