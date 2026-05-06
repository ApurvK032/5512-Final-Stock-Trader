"""Bayesian belief updates for a fitted Gaussian HMM."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Return belief entropy scaled to [0, 1]."""
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-300, 1.0)
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = np.log(len(probs))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


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


def filter_beliefs(
    features: pd.DataFrame,
    model: GaussianHMM,
    initial_belief: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run Bayesian filtering over a sequence of observations."""
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
    belief_array = beliefs[columns].to_numpy()
    entropy = -np.sum(np.clip(belief_array, 1e-300, 1.0) * np.log(np.clip(belief_array, 1e-300, 1.0)), axis=1)
    beliefs["belief_entropy"] = entropy
    beliefs["normalized_entropy"] = entropy / np.log(n_states)
    beliefs["most_likely_state"] = [
        f"state_{int(i)}" for i in np.argmax(belief_array, axis=1)
    ]
    return beliefs
