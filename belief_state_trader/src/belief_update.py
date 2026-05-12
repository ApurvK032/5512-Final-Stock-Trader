"""Bayesian belief updates for a fitted Gaussian HMM."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def observation_log_likelihood(model: GaussianHMM, observation: np.ndarray) -> np.ndarray:
    """Return log P(observation | state) for every hidden state."""
    log_probs = np.empty(model.n_components)
    for state in range(model.n_components):
        log_probs[state] = multivariate_normal.logpdf(
            observation,
            mean=model.means_[state],
            cov=model.covars_[state],
            allow_singular=True,
        )
    return log_probs


def belief_entropy(belief: np.ndarray) -> float:
    """Shannon entropy H(b) = -sum(b(s) * log(b(s))), using natural log."""
    b = np.clip(belief, 1e-300, None)
    return float(-np.sum(b * np.log(b)))


def filter_beliefs(
    features: pd.DataFrame,
    model: GaussianHMM,
    initial_belief: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run Bayesian filtering over a sequence of observations.

    Returns a DataFrame with per-state probabilities, most likely state,
    and belief entropy for each time step.
    """
    n_states = model.n_components
    belief = (
        np.full(n_states, 1.0 / n_states)
        if initial_belief is None
        else np.asarray(initial_belief, dtype=float)
    )
    belief = belief / belief.sum()

    rows = []
    for _, row in features.iterrows():
        predicted = model.transmat_.T @ belief
        predicted = np.clip(predicted, 1e-300, None)

        log_post = np.log(predicted) + observation_log_likelihood(
            model, row.to_numpy(dtype=float)
        )
        belief = np.exp(log_post - logsumexp(log_post))
        rows.append(belief.copy())

    columns = [f"state_{i}_prob" for i in range(n_states)]
    beliefs = pd.DataFrame(rows, index=features.index, columns=columns)
    beliefs["most_likely_state"] = [
        f"state_{int(i)}" for i in np.argmax(beliefs[columns].to_numpy(), axis=1)
    ]
    beliefs["entropy"] = [
        belief_entropy(beliefs.loc[idx, columns].to_numpy(dtype=float))
        for idx in beliefs.index
    ]
    max_entropy = float(np.log(n_states))
    beliefs["normalized_entropy"] = beliefs["entropy"] / max_entropy if max_entropy > 0 else 0.0
    return beliefs

