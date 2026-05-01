"""HMM fitting helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class FittedHMM:
    """Fitted HMM plus training metadata."""

    model: GaussianHMM
    feature_columns: list[str]
    best_seed: int
    log_likelihood: float


def fit_gaussian_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    n_seeds: int = 10,
    base_seed: int = 42,
    n_iter: int = 200,
) -> FittedHMM:
    """Fit a Gaussian HMM and keep the best random restart by log-likelihood."""
    X = features.to_numpy(dtype=float)
    best_model = None
    best_log_likelihood = -np.inf
    best_seed = base_seed

    for offset in range(n_seeds):
        seed = base_seed + offset
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
        )
        model.fit(X)
        log_likelihood = float(model.score(X))

        if log_likelihood > best_log_likelihood:
            best_model = model
            best_log_likelihood = log_likelihood
            best_seed = seed

    if best_model is None:
        raise RuntimeError("HMM fitting failed for all random seeds.")

    return FittedHMM(
        model=best_model,
        feature_columns=list(features.columns),
        best_seed=best_seed,
        log_likelihood=best_log_likelihood,
    )


def decoded_state_summary(
    fitted: FittedHMM,
    features: pd.DataFrame,
    real_log_returns: pd.Series,
) -> pd.DataFrame:
    """Summarize HMM states after Viterbi decoding on training data."""
    valid = real_log_returns.notna()
    features = features.loc[valid]
    real_log_returns = real_log_returns.loc[valid]
    X = features.to_numpy(dtype=float)
    states = fitted.model.predict(X)

    rows = []
    for state in range(fitted.model.n_components):
        mask = states == state
        state_returns = real_log_returns.iloc[mask].dropna()
        rows.append(
            {
                "state": state,
                "n_days": int(mask.sum()),
                "state_frequency": float(mask.mean()),
                "mean_real_log_return": float(state_returns.mean()),
                "annualized_mean_return": float(state_returns.mean() * 252),
                "annualized_volatility": float(state_returns.std(ddof=1) * np.sqrt(252)),
            }
        )

    return pd.DataFrame(rows).sort_values("annualized_mean_return").reset_index(drop=True)


def transition_matrix_frame(fitted: FittedHMM) -> pd.DataFrame:
    """Return the learned transition matrix as a labeled DataFrame."""
    labels = [f"state_{i}" for i in range(fitted.model.n_components)]
    return pd.DataFrame(fitted.model.transmat_, index=labels, columns=labels)
