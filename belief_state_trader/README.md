# Belief State Trader

Algorithm pipeline for the CSCI 5512 final project.

The project models S&P 500 trading as a decision problem under uncertainty. The
main idea is to learn hidden market regimes from historical data, then later use
Bayesian belief updates to reason about which regime the market is likely in.

Current status: the data pipeline, data checks, real-return calculation, three
planned baselines, and basic HMM parameter estimation are implemented.

## Current Workflow

```text
S&P 500 CSV files
    |
    v
load train/test data
    |
    v
check data quality and feature statistics
    |
    v
compute real log returns from Adj_Close
    |
    v
run three planned baselines
    |
    v
fit K=3 Gaussian HMM on training features
```

## Folder Structure

```text
belief_state_trader/
|-- data/        # train/test CSV files
|-- notes/       # explanation notes and workflow overview
|-- results/     # generated summaries and fitted HMM output
|-- scripts/     # runnable project steps
|-- src/         # reusable Python modules
|-- README.md
```

## Data

Local data files:

```text
data/sp500_train.csv
data/sp500_test.csv
```

Current data split:

| Split | Date Range | Rows |
|---|---|---:|
| Training | 2015-01-02 to 2020-12-31 | 1511 |
| Test | 2021-01-04 to 2025-12-30 | 1254 |

Model feature columns:

```text
Log_Return
Volume
Volatility
```

For backtesting/performance calculations, the pipeline computes:

```text
Real_Log_Return = log(Adj_Close_t / Adj_Close_{t-1})
```

This is separate from the CSV's `Log_Return` column, which appears to be a
standardized model feature.

## Implemented Baselines

The original project plan includes three baselines. All three are implemented.

| Baseline | Meaning |
|---|---|
| Buy and Hold | Always fully invested in the S&P 500. |
| Moving Average Crossover 50/200 | Invest when 50-day moving average is above 200-day moving average. |
| Single-Regime No-Belief | Uses one training-set mean/variance estimate and one fixed test-period weight. |

Current test-period results:

| Strategy | Total Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Buy and Hold | 85.04% | 0.739 | -25.43% |
| Moving Average Crossover 50/200 | 33.86% | 0.486 | -19.74% |
| Single-Regime No-Belief | 55.12% | 0.739 | -18.88% |

## HMM Parameter Estimation

The first algorithm task from the project note is implemented:

```text
Implement HMM parameter estimation (hmmlearn)
```

Current HMM setup:

```text
library: hmmlearn
model: GaussianHMM
hidden states: K = 3
features: Log_Return, Volume, Volatility
training split only
```

Current HMM training result:

```text
best seed: 42
training log-likelihood: -3497.4771
```

Learned states sorted by annualized mean return:

| State | Frequency | Annualized Mean Return | Annualized Volatility | Interpretation |
|---|---:|---:|---:|---|
| 0 | 4.0% | -70.87% | 64.71% | Bear-like/high-risk state |
| 1 | 31.3% | 6.86% | 20.69% | Sideways/moderate state |
| 2 | 64.7% | 16.55% | 9.06% | Bull-like/lower-risk state |

## How To Run

Run these commands from the `belief_state_trader` folder.

Data summary:

```bash
python scripts/01_data_summary.py
```

Baselines:

```bash
python scripts/02_buy_hold_baseline.py
python scripts/03_moving_average_baseline.py
python scripts/04_single_regime_baseline.py
```

HMM fitting:

```bash
python scripts/05_fit_hmm.py
```

## Main Files

Reusable modules:

```text
src/data.py       # data loading and real-return calculation
src/backtest.py   # equity curve and performance metrics
src/baselines.py  # baseline strategy logic
src/hmm_model.py  # Gaussian HMM fitting helpers
```

Runnable scripts:

```text
scripts/01_data_summary.py
scripts/02_buy_hold_baseline.py
scripts/03_moving_average_baseline.py
scripts/04_single_regime_baseline.py
scripts/05_fit_hmm.py
```

Important outputs:

```text
results/buy_hold_summary.csv
results/moving_average_summary.csv
results/single_regime_summary.csv
results/hmm_model.pkl
results/hmm_state_summary.csv
results/hmm_transition_matrix.csv
results/hmm_training_summary.txt
```

## Notes

Start here for the team-facing explanation:

```text
notes/workflow_overview.md
```

Additional notes:

```text
notes/01_project_understanding.md
notes/02_data_understanding.md
notes/03_baselines_and_backtesting.md
notes/04_hmm_understanding.md
```

## Next Step

The next algorithm task is:

```text
Implement Bayesian belief updates
```

This will use the learned HMM transition matrix and observation likelihoods to
update the probability of each hidden market regime over time.
