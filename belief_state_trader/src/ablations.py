"""Ablation studies for the Bayesian belief-state trading agent."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src import backtest, belief_update, data, hmm_model, portfolio


# ---------------------------------------------------------------------------
# 1. Risk aversion sweep
# ---------------------------------------------------------------------------

def risk_aversion_sweep(
    fitted: hmm_model.FittedHMM,
    train: pd.DataFrame,
    test: pd.DataFrame,
    lambdas: list[float] | None = None,
) -> pd.DataFrame:
    """Backtest the Bayesian strategy at different risk aversion levels."""
    if lambdas is None:
        lambdas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    train_features = data.feature_matrix(train, fitted.feature_columns)
    valid = train["Real_Log_Return"].notna()
    train_states = fitted.model.predict(train_features.loc[valid].to_numpy(dtype=float))
    state_moments = portfolio.estimate_state_return_moments(
        train_states, train.loc[valid, "Real_Log_Return"], fitted.model.n_components,
    )

    test_features = data.feature_matrix(test, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(test_features, fitted.model)

    rows = []
    for lam in lambdas:
        weights = portfolio.weights_from_beliefs(beliefs, state_moments, risk_aversion=lam)
        returns = portfolio.returns_from_weights(weights, test["Real_Log_Return"])
        summary = backtest.summarize_backtest(f"lambda={lam}", returns)
        summary["risk_aversion"] = lam
        summary["average_weight"] = float(weights.mean())
        rows.append(summary)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Feature ablations
# ---------------------------------------------------------------------------

FEATURE_SUBSETS = {
    "returns_only": ["Log_Return"],
    "returns_volume": ["Log_Return", "Volume"],
    "returns_volatility": ["Log_Return", "Volatility"],
    "all_features": ["Log_Return", "Volume", "Volatility"],
}


def feature_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_states: int = 3,
    risk_aversion: float = 2.0,
) -> pd.DataFrame:
    """Fit separate HMMs for different feature subsets and compare."""
    rows = []
    for subset_name, columns in FEATURE_SUBSETS.items():
        train_features = data.feature_matrix(train_df, columns)
        fitted = hmm_model.fit_gaussian_hmm(train_features, n_states=n_states)

        valid = train_df["Real_Log_Return"].notna()
        train_states = fitted.model.predict(
            train_features.loc[valid].to_numpy(dtype=float)
        )
        state_moments = portfolio.estimate_state_return_moments(
            train_states, train_df.loc[valid, "Real_Log_Return"],
            fitted.model.n_components,
        )

        test_features = data.feature_matrix(test_df, columns)
        beliefs = belief_update.filter_beliefs(test_features, fitted.model)
        weights = portfolio.weights_from_beliefs(beliefs, state_moments, risk_aversion)
        returns = portfolio.returns_from_weights(weights, test_df["Real_Log_Return"])

        summary = backtest.summarize_backtest(subset_name, returns)
        summary["features"] = ", ".join(columns)
        summary["n_features"] = len(columns)
        summary["average_weight"] = float(weights.mean())
        summary["hmm_log_likelihood"] = fitted.log_likelihood
        rows.append(summary)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Entropy-aware vs fixed-confidence position sizing
# ---------------------------------------------------------------------------

def entropy_scaled_weights(
    beliefs: pd.DataFrame,
    state_moments: pd.DataFrame,
    risk_aversion: float = 2.0,
    entropy_penalty: float = 1.0,
) -> pd.Series:
    """Scale portfolio weights by (1 - normalized_entropy).

    When entropy is high (uncertain), reduce position. When entropy is low
    (confident), keep the mean-variance weight.
    """
    base_weights = portfolio.weights_from_beliefs(beliefs, state_moments, risk_aversion)
    confidence = 1.0 - beliefs["normalized_entropy"].clip(0, 1)
    scaled = base_weights * (confidence ** entropy_penalty)
    return scaled.clip(0.0, 1.0).rename("weight")


def entropy_ablation(
    fitted: hmm_model.FittedHMM,
    train: pd.DataFrame,
    test: pd.DataFrame,
    risk_aversion: float = 2.0,
) -> pd.DataFrame:
    """Compare: standard mean-var weights vs entropy-scaled weights."""
    train_features = data.feature_matrix(train, fitted.feature_columns)
    valid = train["Real_Log_Return"].notna()
    train_states = fitted.model.predict(train_features.loc[valid].to_numpy(dtype=float))
    state_moments = portfolio.estimate_state_return_moments(
        train_states, train.loc[valid, "Real_Log_Return"], fitted.model.n_components,
    )

    test_features = data.feature_matrix(test, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(test_features, fitted.model)

    rows = []

    # Standard mean-variance weights (no entropy scaling)
    std_weights = portfolio.weights_from_beliefs(beliefs, state_moments, risk_aversion)
    std_returns = portfolio.returns_from_weights(std_weights, test["Real_Log_Return"])
    summary = backtest.summarize_backtest("Standard (no entropy scaling)", std_returns)
    summary["method"] = "standard"
    summary["average_weight"] = float(std_weights.mean())
    rows.append(summary)

    # Entropy-scaled with different penalty strengths
    for penalty in [0.5, 1.0, 2.0, 3.0]:
        ent_weights = entropy_scaled_weights(
            beliefs, state_moments, risk_aversion, entropy_penalty=penalty,
        )
        ent_returns = portfolio.returns_from_weights(ent_weights, test["Real_Log_Return"])
        summary = backtest.summarize_backtest(
            f"Entropy-scaled (penalty={penalty})", ent_returns,
        )
        summary["method"] = f"entropy_penalty_{penalty}"
        summary["average_weight"] = float(ent_weights.mean())
        rows.append(summary)

    return pd.DataFrame(rows)
