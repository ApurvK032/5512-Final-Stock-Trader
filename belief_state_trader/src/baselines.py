"""Baseline trading strategies.

All baselines produce a daily weight series so that transaction costs
(0.1% per unit of weight change) are applied uniformly via
portfolio.returns_from_weights().
"""
from __future__ import annotations

import pandas as pd


def buy_and_hold_weights(real_log_returns: pd.Series) -> pd.Series:
    """Buy and Hold: weight = 1.0 every day."""
    valid = real_log_returns.dropna()
    return pd.Series(1.0, index=valid.index, name="weight")


def buy_and_hold_returns(real_log_returns: pd.Series) -> pd.Series:
    """Buy and Hold returns (via weight series with transaction costs)."""
    from src.portfolio import returns_from_weights
    weights = buy_and_hold_weights(real_log_returns)
    return returns_from_weights(weights, real_log_returns)


def moving_average_weights(
    prices: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.Series:
    """MA crossover weight: 1.0 when short MA > long MA, else 0.0."""
    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()
    position = (short_ma > long_ma).astype(float)
    return position.dropna().rename("weight")


def moving_average_crossover_returns(
    prices: pd.Series,
    real_log_returns: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.Series:
    """MA crossover returns with transaction costs."""
    from src.portfolio import returns_from_weights
    weights = moving_average_weights(prices, short_window, long_window)
    return returns_from_weights(weights, real_log_returns)


def single_regime_weight(
    train_real_log_returns: pd.Series,
    test_real_log_returns: pd.Series,
    risk_aversion: float = 2.0,
) -> tuple[pd.Series, float]:
    """Compute the single fixed weight from training data and return a weight series."""
    train_returns = train_real_log_returns.dropna()
    test_returns = test_real_log_returns.dropna()

    mean_return = train_returns.mean()
    variance = train_returns.var(ddof=1)
    if variance <= 0:
        w = 0.0
    else:
        w = mean_return / (2.0 * risk_aversion * variance)
    w = float(max(0.0, min(1.0, w)))

    weights = pd.Series(w, index=test_returns.index, name="weight")
    return weights, w


def single_regime_no_belief_returns(
    train_real_log_returns: pd.Series,
    test_real_log_returns: pd.Series,
    risk_aversion: float = 2.0,
) -> tuple[pd.Series, float]:
    """Single-regime returns with transaction costs."""
    from src.portfolio import returns_from_weights
    weights, w = single_regime_weight(
        train_real_log_returns, test_real_log_returns, risk_aversion,
    )
    return returns_from_weights(weights, test_real_log_returns), w
