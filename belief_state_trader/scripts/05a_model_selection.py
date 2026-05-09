"""BIC/AIC model selection sweep for the number of HMM states.

Run from the belief_state_trader folder:

    python scripts/05a_model_selection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data, hmm_model


def main():
    train = data.add_real_log_return(data.load_train())
    features = data.feature_matrix(train)

    print("Running BIC/AIC model selection sweep (K=2..8)...")
    results = hmm_model.model_selection_sweep(
        features, k_range=range(2, 9), n_seeds=10, base_seed=42, n_iter=200
    )

    results_dir = PROJECT_ROOT / "results"
    csv_path = results_dir / "model_selection.csv"
    results.to_csv(csv_path, index=False)

    print("\nModel selection results:")
    print(results.to_string(index=False))
    print(f"\nBest K by BIC: {results.loc[results['bic'].idxmin(), 'n_states']}")
    print(f"Best K by AIC: {results.loc[results['aic'].idxmin(), 'n_states']}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["n_states"], results["bic"], "o-", label="BIC", linewidth=2)
    ax.plot(results["n_states"], results["aic"], "s--", label="AIC", linewidth=2)
    ax.set_xlabel("Number of Hidden States (K)")
    ax.set_ylabel("Information Criterion")
    ax.set_title("HMM Model Selection: BIC and AIC vs. Number of States")
    ax.set_xticks(results["n_states"].tolist())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = results_dir / "model_selection.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved results to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
