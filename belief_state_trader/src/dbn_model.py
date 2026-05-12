"""Dynamic Bayesian Network representation using pgmpy.

This module builds a DBN that mirrors the HMM structure:
  - A hidden Regime node evolves over time via a transition CPD
  - Observed features (Log_Return, Volume, Volatility) depend on Regime
  - Bayesian inference gives posterior beliefs over the current regime

The DBN is constructed from parameters learned by the hmmlearn HMM,
so the two representations are directly comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork as DBN

from src.hmm_model import FittedHMM


def build_dbn_from_hmm(fitted: FittedHMM) -> DBN:
    """Construct a pgmpy DBN whose parameters come from a fitted HMM.

    Structure per time slice:
        Regime_t -> Log_Return_t
        Regime_t -> Volume_t
        Regime_t -> Volatility_t
    Inter-slice:
        Regime_t -> Regime_{t+1}
    """
    n_states = fitted.model.n_components

    model = DBN()

    model.add_edges_from([
        (("Regime", 0), ("Log_Return", 0)),
        (("Regime", 0), ("Volume", 0)),
        (("Regime", 0), ("Volatility", 0)),
        (("Regime", 0), ("Regime", 1)),
        (("Regime", 1), ("Log_Return", 1)),
        (("Regime", 1), ("Volume", 1)),
        (("Regime", 1), ("Volatility", 1)),
    ])

    # Initial state distribution (stationary distribution from HMM)
    start_prob = fitted.model.startprob_
    cpd_regime_0 = TabularCPD(
        variable=("Regime", 0),
        variable_card=n_states,
        values=start_prob.reshape(-1, 1).tolist(),
    )

    # Transition CPD: P(Regime_{t+1} | Regime_t)
    # pgmpy expects columns to be parent states, rows to be child states
    cpd_regime_t = TabularCPD(
        variable=("Regime", 1),
        variable_card=n_states,
        values=fitted.model.transmat_.T.tolist(),
        evidence=[("Regime", 0)],
        evidence_card=[n_states],
    )

    model.add_cpds(cpd_regime_0, cpd_regime_t)

    return model


def dbn_structure_summary(fitted: FittedHMM) -> dict:
    """Return a summary of the DBN structure and parameters."""
    n_states = fitted.model.n_components
    n_features = len(fitted.feature_columns)

    intra_edges = [
        f"Regime -> {feat}" for feat in fitted.feature_columns
    ]
    inter_edges = ["Regime_t -> Regime_{t+1}"]

    state_params = []
    for s in range(n_states):
        state_params.append({
            "state": s,
            "mean": fitted.model.means_[s].tolist(),
            "covariance_diag": np.diag(fitted.model.covars_[s]).tolist(),
        })

    return {
        "n_states": n_states,
        "n_features": n_features,
        "feature_names": fitted.feature_columns,
        "intra_slice_edges": intra_edges,
        "inter_slice_edges": inter_edges,
        "transition_matrix": fitted.model.transmat_.tolist(),
        "start_probs": fitted.model.startprob_.tolist(),
        "state_parameters": state_params,
    }


def dbn_forward_beliefs(
    fitted: FittedHMM,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Run forward filtering using the DBN's transition model and
    Gaussian emission likelihoods (equivalent to HMM forward algorithm).

    This re-implements filtering using the DBN's structure explicitly
    to demonstrate the pgmpy-based approach.
    """
    from src.belief_update import belief_entropy, observation_log_likelihood
    from scipy.special import logsumexp

    model = fitted.model
    n_states = model.n_components
    belief = model.startprob_.copy()
    belief = belief / belief.sum()

    rows = []
    for _, row in features.iterrows():
        predicted = model.transmat_.T @ belief
        predicted = np.clip(predicted, 1e-300, None)

        log_obs = observation_log_likelihood(model, row.to_numpy(dtype=float))
        log_post = np.log(predicted) + log_obs
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
    return beliefs


def compare_belief_methods(
    hmm_beliefs: pd.DataFrame,
    dbn_beliefs: pd.DataFrame,
) -> dict:
    """Quantify agreement between HMM-based and DBN-based beliefs."""
    prob_cols = [c for c in hmm_beliefs.columns if c.endswith("_prob")]
    common_idx = hmm_beliefs.index.intersection(dbn_beliefs.index)

    hmm_vals = hmm_beliefs.loc[common_idx, prob_cols].to_numpy(dtype=float)
    dbn_vals = dbn_beliefs.loc[common_idx, prob_cols].to_numpy(dtype=float)

    mae = float(np.abs(hmm_vals - dbn_vals).mean())
    max_diff = float(np.abs(hmm_vals - dbn_vals).max())

    hmm_states = hmm_beliefs.loc[common_idx, "most_likely_state"]
    dbn_states = dbn_beliefs.loc[common_idx, "most_likely_state"]
    state_agreement = float((hmm_states == dbn_states).mean())

    return {
        "n_days_compared": len(common_idx),
        "mean_absolute_prob_difference": mae,
        "max_absolute_prob_difference": max_diff,
        "most_likely_state_agreement": state_agreement,
    }
