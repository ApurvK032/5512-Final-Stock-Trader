# Belief State Trader

This folder contains the implementation for the CSCI 5512 Group 2 final project, **Bayesian Belief-State Trading Agent for Autonomous Portfolio Management**.

The pipeline models S&P 500 trading as a partially observable decision problem. The true market regime is hidden, so the agent learns a Gaussian HMM from historical data, updates Bayesian beliefs over hidden regimes each day, and uses those beliefs to choose a long-only portfolio weight.

## Workflow Summary

```text
S&P 500 train/test CSVs
    |
    v
data loading and real-return calculation
    |
    v
baseline strategies
    |
    v
K=3 Gaussian HMM regime fitting
    |
    v
Bayesian belief updates on test data
    |
    v
belief-to-portfolio allocation
    |
    v
transaction-cost-aware backtesting
    |
    v
model selection, ablations, pgmpy/DBN check, plots, and report tables
```

## Data Split

| Split | Date Range | Use |
|---|---|---|
| Training | 2015-01-02 to 2020-12-31 | HMM fitting and regime return/risk estimation |
| Main Test | 2021-01-04 to 2025-12-30 | Base strategy and baseline evaluation |
| Validation | 2021-2022 | Enhanced strategy parameter selection |
| Held-out Check | 2023-2025 | Final enhanced strategy validation |

The HMM uses standardized feature columns:

```text
Log_Return
Volume
Volatility
```

Backtesting uses separately computed real log returns from adjusted close prices:

```text
Real_Log_Return = log(Adj_Close_t / Adj_Close_{t-1})
```

## Run

From this folder:

```powershell
python scripts/run_all.py
```

Optional compile check:

```powershell
python -m compileall scripts src
```

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Numbered Scripts

| Script | Purpose |
|---|---|
| `00_data_collection.ipynb` | Data download/reference notebook |
| `01_data_summary.py` | Data checks and summary |
| `02_buy_hold_baseline.py` | Buy-and-Hold baseline |
| `03_moving_average_baseline.py` | Moving Average 50/200 baseline |
| `04_single_regime_baseline.py` | Single-Regime No-Belief baseline |
| `05_fit_hmm.py` | Fit the main K=3 Gaussian HMM |
| `05a_model_selection.py` | Sweep HMM model sizes and covariance options |
| `06_bayesian_belief.py` | Generate daily Bayesian regime beliefs |
| `07_bayesian_strategy.py` | Run the base Bayesian portfolio strategy |
| `08_strategy_comparison.py` | Compare base strategies |
| `09_regime_and_calibration.py` | Regime diagnostics and belief calibration |
| `10_ablation_studies.py` | Feature, entropy, and risk-aversion ablations |
| `11_pgmpy_dbn.py` | `pgmpy` Dynamic Bayesian Network representation check |
| `11a_pgmpy_exploration.py` | Additional pgmpy/HMM belief comparison |
| `12_all_plots.py` | Generate report-ready plots |
| `13_enhanced_agent.py` | Enhanced bear-defense strategy experiments |
| `14_validate_enhanced_strategy.py` | Validation-selected enhanced strategy and held-out check |
| `15_bootstrap_confidence_intervals.py` | Bootstrap Sharpe confidence intervals for held-out returns |
| `run_all.py` | Runs the full pipeline in order |

## Reusable Modules

| Module | Purpose |
|---|---|
| `src/data.py` | Data loading, date parsing, sorting, and real-return calculation |
| `src/backtest.py` | Equity curves, returns, volatility, Sharpe, Sortino, drawdown, Calmar |
| `src/baselines.py` | Baseline trading rules |
| `src/hmm_model.py` | Gaussian HMM fitting helpers |
| `src/belief_update.py` | Log-domain Bayesian filtering and belief entropy |
| `src/portfolio.py` | Belief-to-weight logic and transaction-cost calculations |
| `src/ablations.py` | Feature, entropy, and risk-aversion experiment helpers |
| `src/dbn_model.py` | Dynamic Bayesian Network representation utilities |
| `src/enhanced_agent.py` | Enhanced strategy rules |
| `src/evaluation.py` | Additional evaluation helpers |
| `src/plots.py` | Plot generation |

## Main Results

Base strategy comparison on 2021-2025:

| Strategy | Total Return | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|
| Buy and Hold | 85.04% | 0.729 | 0.983 | -25.43% |
| Moving Average 50/200 | 33.19% | 0.477 | 0.499 | -19.90% |
| Single-Regime No-Belief | 55.12% | 0.729 | 0.983 | -18.88% |
| Bayesian Belief-State | 21.57% | 0.432 | 0.561 | -15.11% |

Held-out enhanced strategy check on 2023-2025:

| Strategy | Total Return | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|
| Enhanced Bear-Defense Bayesian | 60.78% | 1.227 | 1.573 | -15.85% |

## Main Outputs

Important generated files:

```text
results/strategy_comparison.csv
results/enhanced_validation_comparison.csv
results/bootstrap_sharpe_ci.csv
results/model_selection.csv
results/ablation_features.csv
results/ablation_entropy.csv
results/ablation_risk_aversion.csv
results/belief_calibration.csv
results/performance_by_period.csv
results/hmm_vs_dbn_comparison.csv
results/dbn_belief_comparison.csv
results/equity_curves.png
results/enhanced_equity_curves.png
results/drawdown_comparison.png
results/model_selection.png
results/calibration_plot.png
```

Final report PDF:

```text
reports/Final_Report.pdf
```

## Interpretation

The base Bayesian agent is useful as a clean belief-state baseline: it lowers volatility and drawdown but is too conservative and trades too often. The enhanced bear-defense strategy keeps the belief-state idea but uses defensive exposure changes more selectively, which improves held-out risk-adjusted performance.

The project should be framed as a risk-aware belief-state trading system, not as a claim that Bayesian trading always beats Buy-and-Hold in raw return.
