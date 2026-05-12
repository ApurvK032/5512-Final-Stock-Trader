"""Bootstrap held-out Sharpe confidence intervals."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import belief_update, data, enhanced_agent, portfolio


N_BOOTSTRAP = 5_000
RNG_SEED = 5512
TRANSACTION_COST_BPS = 10.0


def load_validation_module():
    """Load the held-out validation helpers."""
    script_path = PROJECT_ROOT / "scripts" / "14_validate_enhanced_strategy.py"
    spec = importlib.util.spec_from_file_location("validate_enhanced_strategy", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validation script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sharpe(values: np.ndarray) -> float:
    """Annualized Sharpe ratio with zero risk-free rate."""
    std = values.std(ddof=1)
    if std == 0:
        return 0.0
    return float(values.mean() / std * np.sqrt(252))


def bootstrap_sharpe_ci(
    returns: pd.Series,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Bootstrap one Sharpe ratio."""
    values = returns.dropna().to_numpy(dtype=float)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = sharpe(sample)

    return {
        "estimate": sharpe(values),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_days": int(len(values)),
    }


def bootstrap_paired_sharpe_diff_ci(
    enhanced_returns: pd.Series,
    buy_hold_returns: pd.Series,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Bootstrap the paired Sharpe difference."""
    common_index = enhanced_returns.dropna().index.intersection(
        buy_hold_returns.dropna().index
    )
    enhanced_values = enhanced_returns.loc[common_index].to_numpy(dtype=float)
    buy_hold_values = buy_hold_returns.loc[common_index].to_numpy(dtype=float)

    boot = np.empty(n_bootstrap)
    n = len(common_index)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = sharpe(enhanced_values[idx]) - sharpe(buy_hold_values[idx])

    return {
        "estimate": sharpe(enhanced_values) - sharpe(buy_hold_values),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_days": int(n),
    }


def selected_enhanced_config(result_dir: Path) -> dict:
    """Read the validation-selected enhanced strategy parameters."""
    selected_path = result_dir / "enhanced_validation_selected.csv"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing {selected_path}. Run scripts/14_validate_enhanced_strategy.py first."
        )

    selected = pd.read_csv(selected_path)
    heldout = selected[selected["evaluation_window"] == "heldout_2023_2025"]
    if heldout.empty:
        raise ValueError(f"No held-out selected row found in {selected_path}")

    row = heldout.iloc[0]
    return {
        "variant": row["variant"],
        "bear_threshold": float(row["bear_threshold"]),
        "defense_scale": float(row["defense_scale"]),
        "min_weight": float(row["min_weight"]),
        "rebalance_threshold": float(row["rebalance_threshold"]),
        "smoothing_alpha": float(row["smoothing_alpha"]),
    }


def heldout_strategy_returns() -> dict[str, pd.Series]:
    """Rebuild held-out daily returns for the baselines and enhanced strategy."""
    validation = load_validation_module()
    result_dir = PROJECT_ROOT / "results"

    train = data.add_real_log_return(data.load_train())
    test = data.add_real_log_return(data.load_test())

    fitted = enhanced_agent.fit_enhanced_hmm(train)
    state_roles = enhanced_agent.identify_state_roles(fitted, train)
    beliefs = belief_update.filter_beliefs(
        data.feature_matrix(test, fitted.feature_columns),
        fitted.model,
    )

    returns_by_strategy: dict[str, pd.Series] = {}
    for name, weights in validation.baseline_weights(train, test).items():
        returns = portfolio.returns_from_weights(
            weights,
            test["Real_Log_Return"],
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
        returns_by_strategy[name] = validation.date_window(
            returns,
            validation.HELDOUT_START,
            validation.HELDOUT_END,
        ).dropna()

    config = selected_enhanced_config(result_dir)
    enhanced_weights = validation.weights_for_config(config, beliefs, state_roles)
    enhanced_returns = portfolio.returns_from_weights(
        enhanced_weights,
        test["Real_Log_Return"],
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    returns_by_strategy["Enhanced Bear-Defense"] = validation.date_window(
        enhanced_returns,
        validation.HELDOUT_START,
        validation.HELDOUT_END,
    ).dropna()

    return returns_by_strategy


def main() -> None:
    result_dir = PROJECT_ROOT / "results"
    result_dir.mkdir(exist_ok=True)

    returns_by_strategy = heldout_strategy_returns()
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for strategy in [
        "Buy and Hold",
        "Moving Average 50/200",
        "Single-Regime No-Belief",
        "Enhanced Bear-Defense",
    ]:
        ci = bootstrap_sharpe_ci(returns_by_strategy[strategy], rng)
        rows.append(
            {
                "metric": "sharpe",
                "strategy": strategy,
                **ci,
                "n_bootstrap": N_BOOTSTRAP,
                "seed": RNG_SEED,
            }
        )

    diff_ci = bootstrap_paired_sharpe_diff_ci(
        returns_by_strategy["Enhanced Bear-Defense"],
        returns_by_strategy["Buy and Hold"],
        rng,
    )
    rows.append(
        {
            "metric": "sharpe_difference",
            "strategy": "Enhanced Bear-Defense minus Buy and Hold",
            **diff_ci,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": RNG_SEED,
        }
    )

    output = pd.DataFrame(rows)
    output_path = result_dir / "bootstrap_sharpe_ci.csv"
    output.to_csv(output_path, index=False)

    print("Bootstrap Sharpe confidence intervals")
    print("-------------------------------------")
    print(output.round(4).to_string(index=False))
    print(f"\nSaved bootstrap results to: {output_path}")


if __name__ == "__main__":
    main()
