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


def hmm_n_free_params(n_states: int, n_features: int) -> int:
    """Count free parameters in a full-covariance Gaussian HMM."""
    start_probs = n_states - 1
    transitions = n_states * (n_states - 1)
    means = n_states * n_features
    covariances = n_states * n_features * (n_features + 1) // 2
    return start_probs + transitions + means + covariances


def compute_bic_aic(fitted: FittedHMM, n_observations: int) -> dict:
    """Compute BIC and AIC for a fitted HMM."""
    n_features = len(fitted.feature_columns)
    k = hmm_n_free_params(fitted.model.n_components, n_features)
    ll = fitted.log_likelihood
    bic = -2 * ll + k * np.log(n_observations)
    aic = -2 * ll + 2 * k
    return {
        "n_states": fitted.model.n_components,
        "n_params": k,
        "log_likelihood": ll,
        "bic": float(bic),
        "aic": float(aic),
    }


def model_selection_sweep(
    features: pd.DataFrame,
    k_range: range = range(2, 6),
    n_seeds: int = 10,
    base_seed: int = 42,
    n_iter: int = 200,
) -> pd.DataFrame:
    """Fit HMMs for each K in k_range, return BIC/AIC comparison table."""
    n_obs = len(features)
    rows = []
    for k in k_range:
        fitted = fit_gaussian_hmm(features, n_states=k, n_seeds=n_seeds,
                                  base_seed=base_seed, n_iter=n_iter)
        info = compute_bic_aic(fitted, n_obs)
        info["best_seed"] = fitted.best_seed
        rows.append(info)
    return pd.DataFrame(rows)
