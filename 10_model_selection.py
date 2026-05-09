"""Compare HMMs with K=2, 3, and 4 states using AIC and BIC.

Lower AIC/BIC = better balance of fit and complexity.
Results are saved to results/model_selection.csv.

Run from the belief_state_trader folder:

    python scripts/10_model_selection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data, hmm_model


def count_free_params(n_states: int, n_features: int) -> int:
    """Count free parameters in a full-covariance GaussianHMM.

    Components
    ----------
    Transition matrix : n_states * (n_states - 1)   (rows sum to 1)
    Means             : n_states * n_features
    Full covariances  : n_states * n_features * (n_features + 1) / 2
    """
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    covars = n_states * n_features * (n_features + 1) // 2
    return trans + means + covars


def main():
    train = data.add_real_log_return(data.load_train())
    features = data.feature_matrix(train)
    X = features.to_numpy(dtype=float)
    n_obs, n_features = X.shape

    print(f"Training observations : {n_obs}")
    print(f"Features              : {list(features.columns)}")
    print("Fitting K = 2, 3, 4  (10 restarts each) ...\n")

    rows = []
    for k in [2, 3, 4]:
        print(f"  K={k} ... ", end="", flush=True)
        fitted = hmm_model.fit_gaussian_hmm(
            features=features,
            n_states=k,
            n_seeds=10,
            base_seed=42,
            n_iter=200,
        )
        ll = fitted.log_likelihood
        k_params = count_free_params(k, n_features)
        aic = 2 * k_params - 2 * ll
        bic = k_params * np.log(n_obs) - 2 * ll

        # Per-state breakdown via Viterbi decoding
        state_summary = hmm_model.decoded_state_summary(
            fitted=fitted,
            features=features,
            real_log_returns=train["Real_Log_Return"],
        )
        freq_str = ", ".join(
            f"s{row.state}={row.state_frequency:.1%}"
            for row in state_summary.itertuples()
        )

        rows.append(
            {
                "n_states": k,
                "log_likelihood": round(ll, 4),
                "n_params": k_params,
                "AIC": round(aic, 2),
                "BIC": round(bic, 2),
                "best_seed": fitted.best_seed,
                "state_frequencies": freq_str,
            }
        )
        print(f"ll={ll:.2f}  AIC={aic:.2f}  BIC={bic:.2f}  [{freq_str}]")

    df = pd.DataFrame(rows)

    output_path = PROJECT_ROOT / "results" / "model_selection.csv"
    df.to_csv(output_path, index=False)

    best_aic_k = int(df.loc[df["AIC"].idxmin(), "n_states"])
    best_bic_k = int(df.loc[df["BIC"].idxmin(), "n_states"])

    print("\nModel selection results (lower AIC / BIC = better fit/complexity tradeoff)")
    print("--------------------------------------------------------------------------")
    print(df.drop(columns=["state_frequencies"]).to_string(index=False))
    print(f"\nBest K by AIC : {best_aic_k}")
    print(f"Best K by BIC : {best_bic_k}")
    print(
        "\nNote: the pipeline currently uses K=3. "
        "If AIC/BIC favour a different K, consider re-fitting with that value."
    )
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
