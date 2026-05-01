"""Small backtesting helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def equity_curve(log_returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    """Convert daily log returns into a portfolio value curve."""
    return starting_value * np.exp(log_returns.cumsum())


def total_return(equity: pd.Series) -> float:
    """Return total percentage growth as a decimal."""
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(log_returns: pd.Series) -> float:
    """Approximate annualized return from average daily log return."""
    return float(log_returns.mean() * TRADING_DAYS_PER_YEAR)


def annualized_volatility(log_returns: pd.Series) -> float:
    """Annualized volatility from daily log returns."""
    return float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(log_returns: pd.Series) -> float:
    """Simple Sharpe ratio with zero risk-free rate."""
    vol = log_returns.std(ddof=1)
    if vol == 0:
        return 0.0
    return float(log_returns.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough drop in the equity curve."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def summarize_backtest(name: str, log_returns: pd.Series) -> dict:
    """Compute a small summary table for one strategy."""
    eq = equity_curve(log_returns)
    return {
        "strategy": name,
        "start_date": log_returns.index.min().date().isoformat(),
        "end_date": log_returns.index.max().date().isoformat(),
        "n_days": int(len(log_returns)),
        "total_return": total_return(eq),
        "annualized_return": annualized_return(log_returns),
        "annualized_volatility": annualized_volatility(log_returns),
        "sharpe": sharpe_ratio(log_returns),
        "max_drawdown": max_drawdown(eq),
    }
