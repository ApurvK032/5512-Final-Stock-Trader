# Bayesian Belief-State Trading Agent

Bayesian Belief-State Trading Agent is the CSCI 5512 Group 2 final project for autonomous portfolio management on the S&P 500. The agent treats the market regime as hidden, learns a Gaussian Hidden Markov Model (HMM) from 2015-2020 data, updates daily Bayesian beliefs over regimes during 2021-2025, and converts those beliefs into long-only portfolio weights between cash and the S&P 500.

The documentation rule for this repo is simple: this README describes the current submitted pipeline. Temporary drafts, comparison notes, and proposal-era ideas are not the source of truth.

## Current Pipeline

1. Load committed S&P 500 train/test CSV files from `belief_state_trader/data/`.
2. Build return, volume, and volatility features.
3. Fit a 3-state Gaussian HMM on the 2015-2020 training split using 10 random restarts.
4. Interpret the learned states as bear-like, sideways/moderate, and bull-like after fitting.
5. Run daily Bayesian filtering on the 2021-2025 test split.
6. Convert posterior regime beliefs into portfolio weights with a long-only mean-variance rule.
7. Evaluate Buy and Hold, Moving Average 50/200, and Single-Regime No-Belief baselines under the same transaction-cost model.
8. Run calibration checks, stress-period analysis, feature/risk/entropy ablations, and a pgmpy DBN representation check.
9. Select the enhanced bear-defense strategy on 2021-2022 validation data and evaluate it on held-out 2023-2025 data.
10. Generate report CSVs, plots, bootstrap Sharpe confidence intervals, and the final report PDF.

## Repository Layout

- `belief_state_trader/data/`: committed train/test S&P 500 CSV files.
- `belief_state_trader/src/`: reusable implementation modules for data loading, HMM fitting, belief updates, portfolio rules, backtesting, ablations, DBN checks, and plotting.
- `belief_state_trader/scripts/`: numbered runnable pipeline steps. `scripts/run_all.py` runs the full pipeline in order.
- `belief_state_trader/results/`: generated result CSVs, fitted model artifact, plots, and bootstrap confidence interval output.
- `belief_state_trader/reports/`: final compiled report PDF.
- `belief_state_trader/PROJECT_FILE_MAP.md`: concise file-by-file map for reviewers.
- `requirements.txt`: Python dependencies for reproducing the pipeline.

## Requirements

- Python 3.10 or newer.
- Packages listed in `requirements.txt`: `numpy`, `pandas`, `scipy`, `matplotlib`, `hmmlearn`, `pgmpy`, `yfinance`, and `jupyter`.

Setup from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a virtual environment is already active, only the final install command is needed.

## Run The Pipeline

From the project implementation folder:

```powershell
cd belief_state_trader
python scripts/run_all.py
```

Optional syntax check:

```powershell
python -m compileall scripts src
```

The raw Yahoo Finance pull is documented in:

```text
belief_state_trader/scripts/00_data_collection.ipynb
```

The full runner uses the committed train/test CSVs, so the main results do not depend on a live Yahoo download.

## Main Results

Base strategy comparison on the 2021-2025 test split:

| Strategy | Total Return | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|
| Buy and Hold | 85.04% | 0.729 | 0.983 | -25.43% |
| Moving Average 50/200 | 33.19% | 0.477 | 0.499 | -19.90% |
| Single-Regime No-Belief | 55.12% | 0.729 | 0.983 | -18.88% |
| Bayesian Belief-State | 21.57% | 0.432 | 0.561 | -15.11% |

Held-out enhanced strategy check on 2023-2025:

| Strategy | Total Return | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|
| Buy and Hold | 80.33% | 1.308 | 1.726 | -18.90% |
| Enhanced Bear-Defense Bayesian | 60.78% | 1.227 | 1.573 | -15.85% |

The main finding is mixed but useful. The base Bayesian agent lowers volatility and drawdown, but it is too conservative in a mostly bullish test period. The enhanced bear-defense version uses regime beliefs more selectively: it does not beat Buy and Hold on raw held-out return, but it reduces drawdown, lowers volatility, and provides a more defensible risk-control rule.

## Key Outputs

Important result files:

```text
belief_state_trader/results/strategy_comparison.csv
belief_state_trader/results/enhanced_validation_comparison.csv
belief_state_trader/results/bootstrap_sharpe_ci.csv
belief_state_trader/results/belief_calibration.csv
belief_state_trader/results/performance_by_period.csv
belief_state_trader/results/ablation_features.csv
belief_state_trader/results/ablation_entropy.csv
belief_state_trader/results/ablation_risk_aversion.csv
belief_state_trader/results/hmm_vs_dbn_comparison.csv
```

Important plots:

```text
belief_state_trader/results/equity_curves.png
belief_state_trader/results/drawdown_comparison.png
belief_state_trader/results/belief_probabilities.png
belief_state_trader/results/entropy_and_weights.png
belief_state_trader/results/enhanced_equity_curves.png
belief_state_trader/results/model_selection_detailed.png
belief_state_trader/results/calibration_plot.png
```

Final report:

```text
belief_state_trader/reports/belief_state_trader_group2_report.pdf
```

## Reproducibility Notes

- Main HMM training uses seeds 42 through 51 and keeps the highest training log-likelihood.
- The base HMM and Bayesian strategy use 2015-2020 for training and 2021-2025 for testing.
- The enhanced strategy is selected only on 2021-2022 validation data, then checked once on held-out 2023-2025 data.
- Bootstrap Sharpe confidence intervals use 5,000 nonparametric resamples with seed 5512.
- The pgmpy DBN component is a structural representation check for the HMM filter, not a replacement trading model.

## Common Commands

Run one specific step:

```powershell
python scripts/05_fit_hmm.py
python scripts/14_validate_enhanced_strategy.py
python scripts/15_bootstrap_confidence_intervals.py
```

Regenerate plots after result CSVs already exist:

```powershell
python scripts/12_all_plots.py
```

Check syntax:

```powershell
python -m compileall scripts src
```

## Notes For Reviewers

This is an educational research pipeline, not a production trading system or financial advice. The useful claim is not that Bayesian regime modeling always maximizes raw return. The useful claim is that explicit beliefs over hidden regimes give a principled way to measure uncertainty, reduce downside exposure, and analyze risk-aware behavior against non-belief baselines.
