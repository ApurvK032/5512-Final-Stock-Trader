"""Run HMM model selection using AIC and BIC.

This script compares Gaussian HMMs with different numbers of hidden states.
It is intended to help justify the final choice of K for the trading model.

Lower AIC/BIC means a better balance between model fit and complexity.

Run from the belief_state_trader folder:

    python scripts/05a_model_selection.py

Outputs:
    results/model_selection.csv
    results/model_selection.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data, hmm_model


K_RANGE = range(2, 9)
N_SEEDS = 10
BASE_SEED = 42
N_ITER = 200


def count_free_params(n_states: int, n_features: int) -> int:
    """Count free parameters in a full-covariance Gaussian HMM.

    Includes:
    - Initial state probabilities: K - 1
    - Transition matrix: K * (K - 1)
    - Means: K * D
    - Full covariance matrices: K * D * (D + 1) / 2

    where K = number of states and D = number of features.
    """
    start_probs = n_states - 1
    transitions = n_states * (n_states - 1)
    means = n_states * n_features
    covariances = n_states * n_features * (n_features + 1) // 2
    return start_probs + transitions + means + covariances


def format_state_frequencies(state_summary: pd.DataFrame) -> str:
    """Create a readable string showing how often each state appears."""
    return ", ".join(
        f"s{int(row.state)}={row.state_frequency:.1%}"
        for row in state_summary.itertuples()
    )


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    train = data.add_real_log_return(data.load_train())
    features = data.feature_matrix(train)
    X = features.to_numpy(dtype=float)
    n_obs, n_features = X.shape

    print("HMM Model Selection")
    print("===================")
    print(f"Training observations : {n_obs}")
    print(f"Feature columns       : {list(features.columns)}")
    print(f"K values tested       : {list(K_RANGE)}")
    print(f"Random restarts       : {N_SEEDS}")
    print(f"Max EM iterations     : {N_ITER}\n")

    rows: list[dict] = []
    failed_k: list[tuple[int, str]] = []

    for k in K_RANGE:
        print(f"Fitting K={k} ... ", end="", flush=True)

        try:
            fitted = hmm_model.fit_gaussian_hmm(
                features=features,
                n_states=k,
                n_seeds=N_SEEDS,
                base_seed=BASE_SEED,
                n_iter=N_ITER,
            )
        except RuntimeError as exc:
            failed_k.append((k, str(exc)))
            print(f"skipped ({exc})")
            continue

        log_likelihood = fitted.log_likelihood
        n_params = count_free_params(k, n_features)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_obs) - 2 * log_likelihood

        state_summary = hmm_model.decoded_state_summary(
            fitted=fitted,
            features=features,
            real_log_returns=train["Real_Log_Return"],
        )
        state_frequencies = format_state_frequencies(state_summary)

        rows.append(
            {
                "n_states": k,
                "log_likelihood": log_likelihood,
                "n_params": n_params,
                "aic": aic,
                "bic": bic,
                "best_seed": fitted.best_seed,
                "state_frequencies": state_frequencies,
            }
        )

        print(
            f"ll={log_likelihood:.2f}, "
            f"AIC={aic:.2f}, "
            f"BIC={bic:.2f}, "
            f"best_seed={fitted.best_seed}, "
            f"[{state_frequencies}]"
        )

    if not rows:
        details = "; ".join(f"K={k}: {msg}" for k, msg in failed_k)
        raise RuntimeError(
            "Model selection failed: all tested K values failed to fit. "
            f"Details: {details}"
        )

    results = pd.DataFrame(rows)

    csv_path = results_dir / "model_selection.csv"
    results.to_csv(csv_path, index=False)

    best_aic_row = results.loc[results["aic"].idxmin()]
    best_bic_row = results.loc[results["bic"].idxmin()]

    print("\nModel selection results")
    print("-----------------------")
    display_cols = [
        "n_states",
        "log_likelihood",
        "n_params",
        "aic",
        "bic",
        "best_seed",
        "state_frequencies",
    ]
    print(results[display_cols].round(4).to_string(index=False))

    print(f"\nBest K by AIC: {int(best_aic_row['n_states'])}")
    print(f"Best K by BIC: {int(best_bic_row['n_states'])}")

    if failed_k:
        print("\nSkipped K values due to numerical fit failures:")
        for k, reason in failed_k:
            print(f"- K={k}: {reason}")

    print(
        "\nInterpretation note:\n"
        "- AIC/BIC may keep improving as K increases.\n"
        "- For the final report, do not choose K only by the lowest number.\n"
        "- Also consider interpretability, state stability, and whether the states map\n"
        "  clearly to market regimes such as bear, sideways, and bull."
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["n_states"], results["bic"], "o-", label="BIC", linewidth=2)
    ax.plot(results["n_states"], results["aic"], "s--", label="AIC", linewidth=2)
    ax.set_xlabel("Number of Hidden States (K)")
    ax.set_ylabel("Information Criterion")
    ax.set_title("HMM Model Selection: AIC/BIC vs. Number of States")
    ax.set_xticks(results["n_states"].tolist())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = results_dir / "model_selection.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved CSV to : {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
