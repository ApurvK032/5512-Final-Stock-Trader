"""Enhanced Bayesian belief-state trading agent.

Improvements over the base agent:
1. Optimal features (Log_Return + Volatility only — Volume hurts)
2. Rebalancing threshold to eliminate unnecessary transaction costs
3. Regime-conditional risk aversion (aggressive in bull, defensive in bear)
4. Weight smoothing via exponential moving average to reduce turnover
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import backtest, belief_update, data, hmm_model, portfolio


def fit_enhanced_hmm(train_df: pd.DataFrame) -> hmm_model.FittedHMM:
    """Fit HMM on optimal feature subset (Log_Return + Volatility)."""
    features = data.feature_matrix(train_df, ["Log_Return", "Volatility"])
    return hmm_model.fit_gaussian_hmm(features, n_states=3, n_seeds=10, n_iter=200)


def identify_state_roles(
    fitted: hmm_model.FittedHMM,
    train_df: pd.DataFrame,
) -> dict:
    """Identify which state is bull, bear, sideways by mean return."""
    features = data.feature_matrix(train_df, fitted.feature_columns)
    valid = train_df["Real_Log_Return"].notna()
    states = fitted.model.predict(features.loc[valid].to_numpy(dtype=float))
    moments = portfolio.estimate_state_return_moments(
        states, train_df.loc[valid, "Real_Log_Return"], fitted.model.n_components,
    )
    sorted_states = moments["mean_return"].sort_values()
    return {
        "bear": int(sorted_states.index[0]),
        "sideways": int(sorted_states.index[1]),
        "bull": int(sorted_states.index[2]),
    }


def dynamic_risk_aversion(
    belief: np.ndarray,
    state_roles: dict,
    normalized_entropy: float,
    bull_lambda: float = 0.8,
    normal_lambda: float = 2.0,
    bear_lambda: float = 8.0,
    bull_confidence: float = 0.75,
    bear_alert: float = 0.10,
) -> float:
    """Adapt risk aversion to the current regime belief.

    - Confident bull: low lambda (aggressive, near full investment)
    - Bear alert: high lambda (defensive, reduce exposure fast)
    - Otherwise: normal lambda
    """
    bull_prob = belief[state_roles["bull"]]
    bear_prob = belief[state_roles["bear"]]

    if bear_prob > bear_alert:
        return bear_lambda
    if bull_prob > bull_confidence and normalized_entropy < 0.15:
        return bull_lambda
    return normal_lambda


def compute_enhanced_weights(
    beliefs: pd.DataFrame,
    state_moments: pd.DataFrame,
    state_roles: dict,
    bull_lambda: float = 0.8,
    normal_lambda: float = 2.0,
    bear_lambda: float = 8.0,
    rebalance_threshold: float = 0.05,
    smoothing_alpha: float = 0.3,
) -> pd.Series:
    """Compute portfolio weights with all enhancements.

    1. Dynamic risk aversion per regime
    2. Weight smoothing (EMA)
    3. Rebalancing threshold
    """
    prob_cols = [f"state_{s}_prob" for s in state_moments.index]

    raw_weights = []
    for idx in beliefs.index:
        b = beliefs.loc[idx, prob_cols].to_numpy(dtype=float)
        norm_ent = beliefs.loc[idx, "normalized_entropy"]
        lam = dynamic_risk_aversion(
            b, state_roles, norm_ent,
            bull_lambda=bull_lambda, normal_lambda=normal_lambda,
            bear_lambda=bear_lambda,
        )
        w = portfolio.optimal_weight(b, state_moments, risk_aversion=lam)
        raw_weights.append(w)

    raw = pd.Series(raw_weights, index=beliefs.index, name="raw_weight")

    # Smooth weights via EMA to reduce turnover
    smoothed = raw.ewm(alpha=smoothing_alpha, adjust=False).mean()

    # Apply rebalancing threshold
    final_weights = []
    current_w = 0.0
    for w in smoothed:
        if abs(w - current_w) > rebalance_threshold:
            current_w = w
        final_weights.append(current_w)

    return pd.Series(final_weights, index=beliefs.index, name="weight").clip(0.0, 1.0)


def run_enhanced_agent(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bull_lambda: float = 0.8,
    normal_lambda: float = 2.0,
    bear_lambda: float = 8.0,
    rebalance_threshold: float = 0.05,
    smoothing_alpha: float = 0.3,
    transaction_cost_bps: float = 10.0,
) -> dict:
    """Run the full enhanced agent pipeline and return results."""
    fitted = fit_enhanced_hmm(train_df)
    state_roles = identify_state_roles(fitted, train_df)

    features_train = data.feature_matrix(train_df, fitted.feature_columns)
    valid = train_df["Real_Log_Return"].notna()
    train_states = fitted.model.predict(
        features_train.loc[valid].to_numpy(dtype=float)
    )
    state_moments = portfolio.estimate_state_return_moments(
        train_states, train_df.loc[valid, "Real_Log_Return"],
        fitted.model.n_components,
    )

    features_test = data.feature_matrix(test_df, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(features_test, fitted.model)

    weights = compute_enhanced_weights(
        beliefs, state_moments, state_roles,
        bull_lambda=bull_lambda,
        normal_lambda=normal_lambda,
        bear_lambda=bear_lambda,
        rebalance_threshold=rebalance_threshold,
        smoothing_alpha=smoothing_alpha,
    )

    returns = portfolio.returns_from_weights(
        weights, test_df["Real_Log_Return"],
        transaction_cost_bps=transaction_cost_bps,
    )

    summary = backtest.summarize_backtest("Enhanced Bayesian Agent", returns)
    summary["average_weight"] = float(weights.mean())
    summary["min_weight"] = float(weights.min())
    summary["max_weight"] = float(weights.max())

    weight_changes = weights.diff().abs().dropna()
    summary["total_turnover"] = float(weight_changes.sum())
    summary["transaction_cost_drag"] = float(weight_changes.sum() * transaction_cost_bps / 10000)
    summary["n_trades"] = int((weight_changes > 0).sum())

    return {
        "summary": summary,
        "weights": weights,
        "returns": returns,
        "beliefs": beliefs,
        "fitted": fitted,
        "state_moments": state_moments,
        "state_roles": state_roles,
    }


def regime_switch_weights(
    beliefs: pd.DataFrame,
    state_roles: dict,
    bull_weight: float = 1.0,
    sideways_weight: float = 0.5,
    bear_weight: float = 0.0,
    bull_threshold: float = 0.5,
    bear_threshold: float = 0.10,
    rebalance_threshold: float = 0.05,
    smoothing_alpha: float = 0.25,
) -> pd.Series:
    """Regime-threshold allocation: stay fully invested during bull,
    exit during bear, partial during sideways.

    This captures most of the bull upside while providing crisis protection.
    """
    bull_col = f"state_{state_roles['bull']}_prob"
    bear_col = f"state_{state_roles['bear']}_prob"

    raw = []
    for idx in beliefs.index:
        bp = beliefs.loc[idx, bull_col]
        brp = beliefs.loc[idx, bear_col]
        if brp > bear_threshold:
            raw.append(bear_weight)
        elif bp > bull_threshold:
            raw.append(bull_weight)
        else:
            raw.append(sideways_weight)

    raw_series = pd.Series(raw, index=beliefs.index)
    smoothed = raw_series.ewm(alpha=smoothing_alpha, adjust=False).mean()

    final = []
    current_w = 0.0
    for w in smoothed:
        if abs(w - current_w) > rebalance_threshold:
            current_w = w
        final.append(current_w)

    return pd.Series(final, index=beliefs.index, name="weight").clip(0.0, 1.0)


def run_regime_switch_agent(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bull_weight: float = 1.0,
    sideways_weight: float = 0.5,
    bear_weight: float = 0.0,
    bull_threshold: float = 0.5,
    bear_threshold: float = 0.10,
    rebalance_threshold: float = 0.05,
    smoothing_alpha: float = 0.25,
    transaction_cost_bps: float = 10.0,
    label: str = "Regime-Switch Bayesian",
) -> dict:
    """Run the regime-switching variant of the enhanced agent."""
    fitted = fit_enhanced_hmm(train_df)
    state_roles = identify_state_roles(fitted, train_df)

    features_test = data.feature_matrix(test_df, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(features_test, fitted.model)

    weights = regime_switch_weights(
        beliefs, state_roles,
        bull_weight=bull_weight,
        sideways_weight=sideways_weight,
        bear_weight=bear_weight,
        bull_threshold=bull_threshold,
        bear_threshold=bear_threshold,
        rebalance_threshold=rebalance_threshold,
        smoothing_alpha=smoothing_alpha,
    )

    returns = portfolio.returns_from_weights(
        weights, test_df["Real_Log_Return"],
        transaction_cost_bps=transaction_cost_bps,
    )

    summary = backtest.summarize_backtest(label, returns)
    summary["average_weight"] = float(weights.mean())
    weight_changes = weights.diff().abs().dropna()
    summary["total_turnover"] = float(weight_changes.sum())
    summary["n_trades"] = int((weight_changes > 0).sum())
    summary["transaction_cost_drag"] = float(weight_changes.sum() * transaction_cost_bps / 10000)

    return {
        "summary": summary,
        "weights": weights,
        "returns": returns,
        "beliefs": beliefs,
        "fitted": fitted,
        "state_roles": state_roles,
    }


def bear_defense_weights(
    beliefs: pd.DataFrame,
    state_roles: dict,
    bear_threshold: float = 0.15,
    defense_scale: float = 3.0,
    min_weight: float = 0.0,
    rebalance_threshold: float = 0.03,
    smoothing_alpha: float = 0.35,
) -> pd.Series:
    """Stay fully invested, reduce proportionally only when bear probability is elevated.

    weight = max(min_weight, 1.0 - defense_scale * max(0, bear_prob - bear_threshold))

    This captures nearly all bull upside while smoothly reducing exposure during bears.
    """
    bear_col = f"state_{state_roles['bear']}_prob"

    raw = []
    for idx in beliefs.index:
        brp = beliefs.loc[idx, bear_col]
        excess = max(0.0, brp - bear_threshold)
        w = max(min_weight, 1.0 - defense_scale * excess)
        raw.append(w)

    raw_series = pd.Series(raw, index=beliefs.index)
    smoothed = raw_series.ewm(alpha=smoothing_alpha, adjust=False).mean()

    final = []
    current_w = 1.0
    for w in smoothed:
        if abs(w - current_w) > rebalance_threshold:
            current_w = w
        final.append(current_w)

    return pd.Series(final, index=beliefs.index, name="weight").clip(0.0, 1.0)


def run_bear_defense_agent(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bear_threshold: float = 0.15,
    defense_scale: float = 3.0,
    min_weight: float = 0.0,
    rebalance_threshold: float = 0.03,
    smoothing_alpha: float = 0.35,
    transaction_cost_bps: float = 10.0,
    label: str = "Bear-Defense Bayesian",
) -> dict:
    """Run the bear-defense variant: fully invested by default, proportional reduction during bears."""
    fitted = fit_enhanced_hmm(train_df)
    state_roles = identify_state_roles(fitted, train_df)

    features_test = data.feature_matrix(test_df, fitted.feature_columns)
    beliefs = belief_update.filter_beliefs(features_test, fitted.model)

    weights = bear_defense_weights(
        beliefs, state_roles,
        bear_threshold=bear_threshold,
        defense_scale=defense_scale,
        min_weight=min_weight,
        rebalance_threshold=rebalance_threshold,
        smoothing_alpha=smoothing_alpha,
    )

    returns = portfolio.returns_from_weights(
        weights, test_df["Real_Log_Return"],
        transaction_cost_bps=transaction_cost_bps,
    )

    summary = backtest.summarize_backtest(label, returns)
    summary["average_weight"] = float(weights.mean())
    weight_changes = weights.diff().abs().dropna()
    summary["total_turnover"] = float(weight_changes.sum())
    summary["n_trades"] = int((weight_changes > 0).sum())
    summary["transaction_cost_drag"] = float(weight_changes.sum() * transaction_cost_bps / 10000)

    return {
        "summary": summary,
        "weights": weights,
        "returns": returns,
        "beliefs": beliefs,
        "fitted": fitted,
        "state_roles": state_roles,
    }


def full_sweep(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Sweep mean-variance, regime-switch, and bear-defense configurations."""
    rows = []

    # Mean-variance enhanced configs
    mv_configs = [
        {"bull_lambda": 0.5, "bear_lambda": 12.0, "rebalance_threshold": 0.08, "smoothing_alpha": 0.2},
        {"bull_lambda": 0.3, "bear_lambda": 10.0, "rebalance_threshold": 0.05, "smoothing_alpha": 0.2},
    ]
    for cfg in mv_configs:
        result = run_enhanced_agent(train_df, test_df, **cfg)
        s = result["summary"]
        s["variant"] = "mean-variance"
        rows.append(s)

    # Regime-switch configs (including aggressive near-full-investment configs)
    rs_configs = [
        {"bull_threshold": 0.5, "bear_threshold": 0.10, "sideways_weight": 0.7,
         "rebalance_threshold": 0.05, "smoothing_alpha": 0.25,
         "label": "RS: bull>0.5, bear>0.10, sw=0.7"},
        {"bull_threshold": 0.3, "bear_threshold": 0.10, "sideways_weight": 0.8,
         "rebalance_threshold": 0.05, "smoothing_alpha": 0.20,
         "label": "RS: bull>0.3, bear>0.10, sw=0.8"},
        {"bull_threshold": 0.3, "bear_threshold": 0.10, "sideways_weight": 0.95,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.30,
         "label": "RS: bull>0.3, bear>0.10, sw=0.95"},
        {"bull_threshold": 0.3, "bear_threshold": 0.15, "sideways_weight": 0.95,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.30,
         "label": "RS: bull>0.3, bear>0.15, sw=0.95"},
        {"bull_threshold": 0.3, "bear_threshold": 0.15, "sideways_weight": 1.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.35,
         "label": "RS: bull>0.3, bear>0.15, sw=1.0"},
        {"bull_threshold": 0.3, "bear_threshold": 0.20, "sideways_weight": 1.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.35,
         "label": "RS: bull>0.3, bear>0.20, sw=1.0"},
    ]
    for cfg in rs_configs:
        result = run_regime_switch_agent(train_df, test_df, **cfg)
        s = result["summary"]
        s["variant"] = "regime-switch"
        rows.append(s)

    # Bear-defense overlay configs (fully invested by default)
    bd_configs = [
        {"bear_threshold": 0.10, "defense_scale": 3.0, "min_weight": 0.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.30,
         "label": "BD: thr=0.10, scale=3.0"},
        {"bear_threshold": 0.15, "defense_scale": 3.0, "min_weight": 0.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.30,
         "label": "BD: thr=0.15, scale=3.0"},
        {"bear_threshold": 0.15, "defense_scale": 4.0, "min_weight": 0.1,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.35,
         "label": "BD: thr=0.15, scale=4.0, min=0.1"},
        {"bear_threshold": 0.20, "defense_scale": 3.0, "min_weight": 0.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.35,
         "label": "BD: thr=0.20, scale=3.0"},
        {"bear_threshold": 0.20, "defense_scale": 4.0, "min_weight": 0.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.35,
         "label": "BD: thr=0.20, scale=4.0"},
        {"bear_threshold": 0.20, "defense_scale": 5.0, "min_weight": 0.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.35,
         "label": "BD: thr=0.20, scale=5.0"},
        {"bear_threshold": 0.25, "defense_scale": 3.0, "min_weight": 0.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.35,
         "label": "BD: thr=0.25, scale=3.0"},
        {"bear_threshold": 0.25, "defense_scale": 4.0, "min_weight": 0.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.35,
         "label": "BD: thr=0.25, scale=4.0"},
        {"bear_threshold": 0.25, "defense_scale": 5.0, "min_weight": 0.0,
         "rebalance_threshold": 0.03, "smoothing_alpha": 0.40,
         "label": "BD: thr=0.25, scale=5.0"},
    ]
    for cfg in bd_configs:
        result = run_bear_defense_agent(train_df, test_df, **cfg)
        s = result["summary"]
        s["variant"] = "bear-defense"
        rows.append(s)

    return pd.DataFrame(rows)
