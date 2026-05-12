# Project File Map

This file summarizes the final project structure for teammates, graders, and future readers.

## Top-Level

```text
belief_state_trader/
|-- data/
|-- reports/
|-- results/
|-- scripts/
|-- src/
|-- PROJECT_FILE_MAP.md
|-- README.md
```

## `data/`

Contains the local S&P 500 CSV files used by the pipeline.

| File | Purpose |
|---|---|
| `sp500_train.csv` | Training period data, 2015-2020 |
| `sp500_test.csv` | Test period data, 2021-2025 |

## `src/`

Reusable Python modules. These contain the implementation logic that scripts call.

| File | Purpose |
|---|---|
| `data.py` | Loads CSV data, parses dates, sorts rows, and computes real returns |
| `backtest.py` | Computes equity curves and metrics such as Sharpe, Sortino, drawdown, and Calmar |
| `baselines.py` | Implements Buy-and-Hold, Moving Average, and Single-Regime baselines |
| `hmm_model.py` | Fits Gaussian HMMs using `hmmlearn` |
| `belief_update.py` | Performs Bayesian filtering and belief entropy calculations |
| `portfolio.py` | Converts beliefs into portfolio weights and applies transaction costs |
| `ablations.py` | Runs feature, entropy, and risk-aversion ablation helpers |
| `dbn_model.py` | Builds the `pgmpy` Dynamic Bayesian Network representation |
| `enhanced_agent.py` | Implements enhanced bear-defense strategy logic |
| `evaluation.py` | Additional evaluation helpers |
| `plots.py` | Creates report-ready plots |

## `scripts/`

Runnable numbered pipeline steps.

| Script | Purpose |
|---|---|
| `00_data_collection.ipynb` | Data collection/reference notebook |
| `01_data_summary.py` | Data checks and summary |
| `02_buy_hold_baseline.py` | Buy-and-Hold baseline |
| `03_moving_average_baseline.py` | Moving Average baseline |
| `04_single_regime_baseline.py` | Single-Regime No-Belief baseline |
| `05_fit_hmm.py` | Main K=3 HMM fitting |
| `05a_model_selection.py` | HMM model-selection sweep |
| `06_bayesian_belief.py` | Daily belief update generation |
| `07_bayesian_strategy.py` | Base Bayesian strategy |
| `08_strategy_comparison.py` | Strategy comparison table |
| `09_regime_and_calibration.py` | Regime diagnostics and calibration |
| `10_ablation_studies.py` | Feature, entropy, and risk-aversion ablations |
| `11_pgmpy_dbn.py` | pgmpy/DBN representation check |
| `11a_pgmpy_exploration.py` | Additional pgmpy/HMM belief comparison |
| `12_all_plots.py` | Plot generation |
| `13_enhanced_agent.py` | Enhanced strategy full-period experiments |
| `14_validate_enhanced_strategy.py` | Validation-selected enhanced strategy and held-out check |
| `15_bootstrap_confidence_intervals.py` | Bootstrap Sharpe confidence intervals for held-out returns |
| `run_all.py` | End-to-end runner |

## `results/`

Generated outputs from the scripts.

Important CSVs:

| File | Purpose |
|---|---|
| `strategy_comparison.csv` | Main base strategy comparison |
| `enhanced_validation_comparison.csv` | Held-out enhanced strategy comparison |
| `enhanced_validation_selected.csv` | Selected enhanced configuration |
| `bootstrap_sharpe_ci.csv` | Bootstrap Sharpe confidence intervals for held-out returns |
| `model_selection.csv` | HMM model-selection results |
| `ablation_features.csv` | Feature ablation results |
| `ablation_entropy.csv` | Entropy strategy ablation results |
| `ablation_risk_aversion.csv` | Risk-aversion sweep results |
| `belief_calibration.csv` | Belief calibration diagnostics |
| `performance_by_period.csv` | Crisis/event-window comparison |
| `hmm_vs_dbn_comparison.csv` | HMM vs DBN belief agreement summary |
| `dbn_belief_comparison.csv` | Supplementary discrete-DBN vs Gaussian-HMM belief comparison |
| `dbn_discrete_beliefs.csv` | Supplementary discrete-DBN belief probabilities |

Important plots:

| File | Purpose |
|---|---|
| `equity_curves.png` | Main base strategy equity curves |
| `enhanced_equity_curves.png` | Enhanced strategy equity curves |
| `drawdown_comparison.png` | Drawdown comparison |
| `model_selection.png` | HMM model-selection plot |
| `calibration_plot.png` | Belief calibration plot |
| `ablation_features.png` | Feature ablation chart |
| `ablation_risk_aversion.png` | Risk-aversion ablation chart |
| `entropy_and_weights.png` | Entropy and portfolio weight behavior |

## `reports/`

Final written report material.

| File | Purpose |
|---|---|
| `belief_state_trader_group2_report.pdf` | Final compiled PDF |
