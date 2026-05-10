"""Advanced evaluation: regime detection, belief calibration, transition analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Drawdown-based crisis period detection
# ---------------------------------------------------------------------------

def compute_drawdown_series(prices: pd.Series) -> pd.Series:
    """Compute running drawdown from peak for a price series."""
    running_max = prices.cummax()
    return prices / running_max - 1.0


def detect_crisis_periods(
    prices: pd.Series,
    threshold: float = -0.10,
) -> pd.DataFrame:
    """Identify contiguous crisis periods where drawdown exceeds threshold."""
    dd = compute_drawdown_series(prices)
    in_crisis = dd < threshold
    periods = []
    start = None
    for date, flag in in_crisis.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            periods.append({
                "start": start,
                "end": date,
                "min_drawdown": float(dd.loc[start:date].min()),
            })
            start = None
    if start is not None:
        periods.append({
            "start": start,
            "end": dd.index[-1],
            "min_drawdown": float(dd.loc[start:].min()),
        })
    return pd.DataFrame(periods)


# ---------------------------------------------------------------------------
# 2. Manual event labels for the test period (2021-2025)
# ---------------------------------------------------------------------------

MANUAL_EVENTS = [
    {
        "name": "2022 Bear Market (Inflation/Rate Hikes)",
        "start": "2022-01-03",
        "end": "2022-10-12",
        "type": "bear",
    },
    {
        "name": "2023 Regional Banking Crisis",
        "start": "2023-03-08",
        "end": "2023-03-27",
        "type": "crisis",
    },
    {
        "name": "2023 Q3 Correction (Bond Yields Spike)",
        "start": "2023-07-31",
        "end": "2023-10-27",
        "type": "correction",
    },
    {
        "name": "2024 August Volatility (Yen Carry Unwind)",
        "start": "2024-07-16",
        "end": "2024-08-05",
        "type": "crisis",
    },
]


def get_manual_events() -> pd.DataFrame:
    """Return manually labeled market events as a DataFrame."""
    df = pd.DataFrame(MANUAL_EVENTS)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def label_dates_by_event(
    dates: pd.DatetimeIndex,
    events: pd.DataFrame,
) -> pd.Series:
    """Return a series labeling each date as the event name or 'normal'."""
    labels = pd.Series("normal", index=dates)
    for _, row in events.iterrows():
        mask = (dates >= row["start"]) & (dates <= row["end"])
        labels[mask] = row["name"]
    return labels


# ---------------------------------------------------------------------------
# 3. Regime detection accuracy: compare HMM states to crisis labels
# ---------------------------------------------------------------------------

def regime_detection_accuracy(
    beliefs: pd.DataFrame,
    event_labels: pd.Series,
    bear_state: int,
) -> dict:
    """Measure how well the HMM's most-likely-state aligns with labeled events.

    bear_state: which HMM state index corresponds to bear/crisis.
    """
    is_crisis = event_labels != "normal"
    hmm_says_bear = beliefs["most_likely_state"] == f"state_{bear_state}"

    aligned = is_crisis.reindex(beliefs.index).fillna(False)
    hmm_aligned = hmm_says_bear.reindex(aligned.index)

    tp = int((hmm_aligned & aligned).sum())
    fp = int((hmm_aligned & ~aligned).sum())
    fn = int((~hmm_aligned & aligned).sum())
    tn = int((~hmm_aligned & ~aligned).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    bear_prob_col = f"state_{bear_state}_prob"
    avg_bear_prob_crisis = float(beliefs.loc[aligned, bear_prob_col].mean()) if aligned.any() else 0.0
    avg_bear_prob_normal = float(beliefs.loc[~aligned, bear_prob_col].mean())

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_bear_prob_during_crisis": avg_bear_prob_crisis,
        "avg_bear_prob_during_normal": avg_bear_prob_normal,
    }


# ---------------------------------------------------------------------------
# 4. Belief calibration: 90% credible interval coverage
# ---------------------------------------------------------------------------

def belief_calibration(
    beliefs: pd.DataFrame,
    state_moments: pd.DataFrame,
    realized_returns: pd.Series,
    confidence_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Check if realized returns fall within credible intervals.

    For each confidence level alpha, compute the predictive interval from
    the belief-weighted mixture and check coverage.
    """
    if confidence_levels is None:
        confidence_levels = [0.50, 0.75, 0.90, 0.95]

    from scipy.stats import norm

    prob_cols = [c for c in beliefs.columns if c.endswith("_prob")]
    means = state_moments["mean_return"].to_numpy(dtype=float)
    variances = state_moments["var_return"].to_numpy(dtype=float)

    common_idx = beliefs.index.intersection(realized_returns.dropna().index)
    results = []

    for alpha in confidence_levels:
        z = norm.ppf(0.5 + alpha / 2.0)
        covered = 0
        total = 0
        for date in common_idx:
            b = beliefs.loc[date, prob_cols].to_numpy(dtype=float)
            pred_mean = float(np.dot(b, means))
            within_var = float(np.dot(b, variances))
            across_var = float(np.dot(b, (means - pred_mean) ** 2))
            pred_std = np.sqrt(within_var + across_var)

            lo = pred_mean - z * pred_std
            hi = pred_mean + z * pred_std
            r = realized_returns.loc[date]
            if lo <= r <= hi:
                covered += 1
            total += 1

        actual_coverage = covered / total if total > 0 else 0.0
        results.append({
            "target_coverage": alpha,
            "actual_coverage": actual_coverage,
            "n_days": total,
            "calibration_error": actual_coverage - alpha,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Regime transition period analysis
# ---------------------------------------------------------------------------

def strategy_performance_by_period(
    strategy_returns: dict[str, pd.Series],
    event_labels: pd.Series,
) -> pd.DataFrame:
    """Compute Sharpe and return for each strategy during each labeled period."""
    from src import backtest

    rows = []
    unique_labels = event_labels.unique()
    for label in unique_labels:
        mask = event_labels == label
        dates = event_labels.index[mask]
        for name, returns in strategy_returns.items():
            period_returns = returns.reindex(dates).dropna()
            if len(period_returns) < 2:
                continue
            rows.append({
                "period": label,
                "strategy": name,
                "n_days": len(period_returns),
                "total_return": float(np.expm1(period_returns.sum())),
                "annualized_return": backtest.annualized_return(period_returns),
                "annualized_volatility": backtest.annualized_volatility(period_returns),
                "sharpe": backtest.sharpe_ratio(period_returns),
                "sortino": backtest.sortino_ratio(period_returns),
                "max_drawdown": backtest.max_drawdown(backtest.equity_curve(period_returns)),
            })

    return pd.DataFrame(rows)


def entropy_during_transitions(
    beliefs: pd.DataFrame,
    event_labels: pd.Series,
) -> pd.DataFrame:
    """Compare belief entropy during crisis vs normal periods."""
    aligned = event_labels.reindex(beliefs.index).fillna("normal")
    rows = []
    for label in aligned.unique():
        mask = aligned == label
        ent = beliefs.loc[mask, "entropy"]
        rows.append({
            "period": label,
            "n_days": int(mask.sum()),
            "mean_entropy": float(ent.mean()),
            "median_entropy": float(ent.median()),
            "std_entropy": float(ent.std()),
            "mean_normalized_entropy": float(beliefs.loc[mask, "normalized_entropy"].mean()),
        })
    return pd.DataFrame(rows)
