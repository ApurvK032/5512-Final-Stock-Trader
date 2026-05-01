"""Baseline trading strategies."""
from __future__ import annotations

import pandas as pd


def buy_and_hold_returns(real_log_returns: pd.Series) -> pd.Series:
    """Buy and Hold: invest 100% in the S&P 500 every day."""
    return real_log_returns.dropna().copy()


def moving_average_crossover_returns(
    prices: pd.Series,
    real_log_returns: pd.Series,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.Series:
    """Invest when the short moving average is above the long moving average."""
    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()
    position = (short_ma > long_ma).astype(float)
    strategy_returns = position.shift(1) * real_log_returns
    return strategy_returns.dropna()


def single_regime_no_belief_returns(
    train_real_log_returns: pd.Series,
    test_real_log_returns: pd.Series,
    risk_aversion: float = 2.0,
) -> tuple[pd.Series, float]:
    """Apply one fixed mean-variance weight estimated from training returns."""
    train_returns = train_real_log_returns.dropna()
    test_returns = test_real_log_returns.dropna()

    mean_return = train_returns.mean()
    variance = train_returns.var(ddof=1)
    if variance <= 0:
        weight = 0.0
    else:
        weight = mean_return / (2.0 * risk_aversion * variance)
    weight = float(max(0.0, min(1.0, weight)))

    return weight * test_returns, weight
