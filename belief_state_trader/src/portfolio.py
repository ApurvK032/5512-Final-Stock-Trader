"""Portfolio decisions from regime beliefs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_state_return_moments(
    states: np.ndarray,
    real_log_returns: pd.Series,
    n_states: int,
) -> pd.DataFrame:
    """Estimate one return mean and variance for each decoded HMM state."""
    returns = real_log_returns.reset_index(drop=True)
    global_mean = returns.mean()
    global_var = returns.var(ddof=1)

    rows = []
    for state in range(n_states):
        state_returns = returns[pd.Series(states) == state]
        if len(state_returns) < 2:
            mean = global_mean
            var = global_var
        else:
            mean = state_returns.mean()
            var = state_returns.var(ddof=1)
        rows.append({"state": state, "mean_return": float(mean), "var_return": float(var)})

    return pd.DataFrame(rows).set_index("state")


def predictive_moments(belief: np.ndarray, state_moments: pd.DataFrame) -> tuple[float, float]:
    """Convert a state belief into predictive mean and variance."""
    mean_by_state = state_moments["mean_return"].to_numpy(dtype=float)
    var_by_state = state_moments["var_return"].to_numpy(dtype=float)

    pred_mean = float(np.dot(belief, mean_by_state))
    within_var = float(np.dot(belief, var_by_state))
    across_var = float(np.dot(belief, (mean_by_state - pred_mean) ** 2))
    return pred_mean, within_var + across_var


def optimal_weight(
    belief: np.ndarray,
    state_moments: pd.DataFrame,
    risk_aversion: float = 2.0,
) -> float:
    """Long-only mean-variance weight clipped to [0, 1]."""
    pred_mean, pred_var = predictive_moments(belief, state_moments)
    if pred_var <= 0:
        return 1.0 if pred_mean > 0 else 0.0
    weight = pred_mean / (2.0 * risk_aversion * pred_var)
    return float(np.clip(weight, 0.0, 1.0))


def weights_from_beliefs(
    beliefs: pd.DataFrame,
    state_moments: pd.DataFrame,
    risk_aversion: float = 2.0,
) -> pd.Series:
    """Compute one portfolio weight for each belief row."""
    prob_columns = [f"state_{state}_prob" for state in state_moments.index]
    weights = [
        optimal_weight(row.to_numpy(dtype=float), state_moments, risk_aversion)
        for _, row in beliefs[prob_columns].iterrows()
    ]
    return pd.Series(weights, index=beliefs.index, name="weight")


def returns_from_weights(
    weights: pd.Series,
    real_log_returns: pd.Series,
    transaction_cost_bps: float = 10.0,
) -> pd.Series:
    """Apply yesterday's weight to today's return, minus transaction costs.

    transaction_cost_bps: cost in basis points per unit of weight change.
    10 bps = 0.1% per trade as specified in the project plan.
    """
    aligned_returns = real_log_returns.reindex(weights.index)
    prev_weights = weights.shift(1).fillna(0.0)
    gross_returns = prev_weights * aligned_returns

    cost_rate = transaction_cost_bps / 10_000.0
    weight_change = (weights - prev_weights).abs()
    costs = weight_change * cost_rate

    strategy_returns = gross_returns - costs
    return strategy_returns.dropna()

