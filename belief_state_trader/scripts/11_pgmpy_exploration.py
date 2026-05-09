"""Explore the HMM as a Dynamic Bayesian Network using pgmpy.

This script does two things:

1. Constructs the equivalent DBN structure in pgmpy, showing how the HMM
   maps to a two-slice temporal Bayesian network (2-TBN):
       Regime_{t-1}  ->  Regime_t   (transition)
       Regime_t      ->  Obs_t      (emission, discretised)

2. Runs a discrete forward filter as a cross-check against the Gaussian
   filter in src/belief_update.py.  Observations are discretised into
   N_BINS quantile bins using training-set thresholds to avoid leakage.

Outputs
-------
results/dbn_belief_comparison.csv  -- per-state Pearson r and mean |diff|
results/dbn_discrete_beliefs.csv   -- raw discrete-filter belief matrix

Run from the belief_state_trader folder:

    python scripts/11_pgmpy_exploration.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import belief_update, data, hmm_model


# pgmpy is optional -- script still runs without it
try:
    from pgmpy.models import DynamicBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


N_BINS = 5   # quantile bins for observation discretisation


# ---------------------------------------------------------------------------
# pgmpy DBN construction (mirrors the fitted HMM topology)
# ---------------------------------------------------------------------------

def build_dbn_from_hmm(fitted: "hmm_model.FittedHMM", emit: np.ndarray, n_bins: int):
    """Return a pgmpy DynamicBayesianNetwork with CPDs from the fitted HMM.

    Topology (pgmpy (variable, time_slice) convention):
        ("Regime", 0) -> ("Regime", 1)   intra-slice Markov transition
        ("Regime", 1) -> ("Obs", 1)      discretised emission
    """
    n_states = fitted.model.n_components
    trans = fitted.model.transmat_   # (K, K)

    dbn = DynamicBayesianNetwork(
        [
            (("Regime", 0), ("Regime", 1)),
            (("Regime", 1), ("Obs", 1)),
        ]
    )

    # P(Regime_t | Regime_{t-1}) -- columns = parent state values
    trans_cpd = TabularCPD(
        variable=("Regime", 1),
        variable_card=n_states,
        values=trans.T,
        evidence=[("Regime", 0)],
        evidence_card=[n_states],
    )

    # P(Obs_t | Regime_t) -- shape (n_bins, n_states)
    emit_cpd = TabularCPD(
        variable=("Obs", 1),
        variable_card=n_bins,
        values=emit.T,
        evidence=[("Regime", 1)],
        evidence_card=[n_states],
    )

    dbn.add_cpds(trans_cpd, emit_cpd)
    return dbn


# ---------------------------------------------------------------------------
# Discrete forward filter (independent of pgmpy)
# ---------------------------------------------------------------------------

def estimate_emit_matrix(
    states: np.ndarray,
    obs_bins: np.ndarray,
    n_states: int,
    n_bins: int,
) -> np.ndarray:
    """Estimate P(obs_bin | state) with Laplace smoothing.  Returns (K, n_bins)."""
    emit = np.ones((n_states, n_bins))   # add-1 prior
    for s in range(n_states):
        for b in range(n_bins):
            emit[s, b] += int(((states == s) & (obs_bins == b)).sum())
        emit[s] /= emit[s].sum()
    return emit


def discrete_forward_filter(
    obs_bins: np.ndarray,
    trans: np.ndarray,
    emit: np.ndarray,
) -> np.ndarray:
    """Standard discrete HMM forward (filtering) pass.  Returns (T, K)."""
    n_states = trans.shape[0]
    T = len(obs_bins)
    beliefs = np.empty((T, n_states))
    belief = np.ones(n_states) / n_states   # uniform prior

    for t in range(T):
        predicted = trans.T @ belief
        updated = predicted * emit[:, obs_bins[t]]
        total = updated.sum()
        belief = updated / total if total > 0 else predicted
        beliefs[t] = belief

    return beliefs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(PROJECT_ROOT / "results" / "hmm_model.pkl", "rb") as f:
        fitted = pickle.load(f)

    train = data.add_real_log_return(data.load_train())
    test  = data.add_real_log_return(data.load_test())
    train_features = data.feature_matrix(train, fitted.feature_columns)
    test_features  = data.feature_matrix(test,  fitted.feature_columns)

    # ------------------------------------------------------------------
    # Discretise Real_Log_Return using training-set quantile edges only
    # ------------------------------------------------------------------
    train_lr = train["Real_Log_Return"].dropna().to_numpy()
    test_lr  = test["Real_Log_Return"].reindex(test_features.index).fillna(0.0).to_numpy()

    edges = np.percentile(train_lr, np.linspace(0, 100, N_BINS + 1))
    edges[0]  = -np.inf
    edges[-1] =  np.inf

    train_bins = np.digitize(train_lr, edges[1:-1])   # values in {0, ..., N_BINS-1}
    test_bins  = np.digitize(test_lr,  edges[1:-1])

    # ------------------------------------------------------------------
    # Discrete emission matrix from training decoded states
    # ------------------------------------------------------------------
    train_states = fitted.model.predict(train_features.to_numpy(dtype=float))
    n_states = fitted.model.n_components
    emit = estimate_emit_matrix(train_states, train_bins, n_states, N_BINS)

    # ------------------------------------------------------------------
    # Discrete forward filter on test
    # ------------------------------------------------------------------
    n_test = min(len(test_bins), len(test_features))
    disc_beliefs = discrete_forward_filter(
        test_bins[:n_test], fitted.model.transmat_, emit
    )

    # ------------------------------------------------------------------
    # Gaussian filter from src/belief_update.py for comparison.
    # belief_update.filter_beliefs returns columns named state_{i}_prob.
    # ------------------------------------------------------------------
    gauss_beliefs = belief_update.filter_beliefs(test_features, fitted.model)
    # Columns are state_0_prob, state_1_prob, state_2_prob
    prob_cols = [f"state_{s}_prob" for s in range(n_states)]
    gauss_arr = gauss_beliefs[prob_cols].to_numpy()[:n_test]

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("DBN / HMM Belief Comparison")
    print("============================")
    print(f"Observation discretisation : {N_BINS} quantile bins (training thresholds)")
    print(f"Hidden states              : {n_states}")
    print(f"Test days compared         : {n_test}")
    print()

    comp_rows = []
    for s in range(n_states):
        corr = float(np.corrcoef(gauss_arr[:, s], disc_beliefs[:, s])[0, 1])
        mad  = float(np.abs(gauss_arr[:, s] - disc_beliefs[:, s]).mean())
        comp_rows.append(
            {
                "state": s,
                "pearson_correlation": round(corr, 4),
                "mean_absolute_difference": round(mad, 4),
            }
        )
        print(f"  State {s}:  correlation = {corr:.4f},  mean |diff| = {mad:.4f}")

    # High correlation confirms the two filters track the same regimes;
    # non-zero MAD reflects the information lost by discretising observations.

    # ------------------------------------------------------------------
    # pgmpy DBN structure
    # ------------------------------------------------------------------
    if PGMPY_AVAILABLE:
        print("\nBuilding pgmpy DBN structure ...")
        dbn = build_dbn_from_hmm(fitted, emit, N_BINS)
        valid = dbn.check_model()
        print(f"  Nodes      : {sorted(dbn.nodes())}")
        print(f"  Edges      : {sorted(dbn.edges())}")
        print(f"  CPDs valid : {valid}")
        print(
            "\n  The DBN mirrors the HMM exactly.  The key advantage of the pgmpy\n"
            "  representation is extensibility: additional observed or latent nodes\n"
            "  (e.g. macro indicators, sector returns) can be wired in as extra\n"
            "  parents of Regime or Obs without changing the core filtering logic."
        )
    else:
        print(
            "\npgmpy not installed -- DBN construction skipped.\n"
            "Install with:  pip install pgmpy"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    results_dir = PROJECT_ROOT / "results"

    comp_df = pd.DataFrame(comp_rows)
    comp_path = results_dir / "dbn_belief_comparison.csv"
    comp_df.to_csv(comp_path, index=False)

    disc_df = pd.DataFrame(
        disc_beliefs,
        columns=[f"discrete_state_{s}_prob" for s in range(n_states)],
    )
    disc_path = results_dir / "dbn_discrete_beliefs.csv"
    disc_df.to_csv(disc_path, index=False)

    print(f"\nSaved belief comparison to : {comp_path}")
    print(f"Saved discrete beliefs to  : {disc_path}")


if __name__ == "__main__":
    main()
