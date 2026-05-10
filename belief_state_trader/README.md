# Belief State Trader

Algorithm pipeline for the CSCI 5512 final project.

The project models S&P 500 trading as a decision problem under uncertainty. The
main idea is to learn hidden market regimes from historical data, then later use
Bayesian belief updates to reason about which regime the market is likely in.

Current status: the data pipeline, data checks, real-return calculation, three
planned baselines, basic HMM parameter estimation, Bayesian belief updates, and
a basic belief-based portfolio strategy are implemented. The scripts now also
produce a combined comparison table and simple result plots.

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
    |
    v
update daily test-period regime beliefs
    |
    v
convert beliefs into portfolio weights
    |
    v
compare all strategies and create plots
```

## Folder Structure

```text
belief_state_trader/
|-- data/        # train/test CSV files
|-- notes/       # explanation notes and workflow overview
|-- results/     # generated summaries, fitted HMM output, and plots
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
| Moving Average Crossover 50/200 | 33.19% | 0.477 | -19.90% |
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

## Bayesian Belief Updates

The second algorithm task from the project note is implemented:

```text
Implement Bayesian belief updates
```

Current belief update setup:

```text
input model: fitted Gaussian HMM
input observations: test-set Log_Return, Volume, Volatility
initial belief: uniform over states
output: daily probability of each hidden state
```

Output:

```text
results/test_beliefs.csv
```

The output contains:

```text
Date
state_0_prob
state_1_prob
state_2_prob
most_likely_state
```

## Belief-Based Strategy

The basic portfolio strategy converts regime beliefs into one daily investment
weight between 0 and 1.

Current setup:

```text
state return/risk estimates: learned from training data
risk aversion: 2.0
weight range: 0 to 1
```

Current test-period result:

| Strategy | Total Return | Sharpe | Max Drawdown | Average Weight |
|---|---:|---:|---:|---:|
| Bayesian Belief-State | 21.57% | 0.432 | -15.11% | 0.611 |

Outputs:

```text
results/bayesian_strategy_summary.csv
results/bayesian_strategy_daily.csv
```

## Strategy Comparison And Plots

The final skeleton step combines the strategy summaries and creates simple
visual outputs.

Outputs:

```text
results/strategy_comparison.csv
results/equity_curves.png
results/belief_probabilities.png
```

## How To Run

Run the full end-to-end skeleton:

```bash
python scripts/run_all.py
```

Or run individual scripts from the `belief_state_trader` folder.


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

Bayesian belief update and strategy:

```bash
python scripts/06_bayesian_belief.py
python scripts/07_bayesian_strategy.py
```

Comparison and plots:

```bash
python scripts/08_strategy_comparison.py
python scripts/09_regime_and_calibration.py
python scripts/10_ablation_studies.py
python scripts/11_pgmpy_dbn.py
python scripts/11a_pgmpy_exploration.py
python scripts/12_all_plots.py
python scripts/13_enhanced_agent.py
```

## Main Files

Reusable modules:

```text
src/data.py       # data loading and real-return calculation
src/backtest.py   # equity curve and performance metrics
src/baselines.py  # baseline strategy logic
src/hmm_model.py  # Gaussian HMM fitting helpers
src/belief_update.py  # daily regime probability updates
src/portfolio.py      # belief-to-weight strategy logic
src/plots.py          # simple result plots
```

Runnable scripts:

```text
scripts/01_data_summary.py
scripts/02_buy_hold_baseline.py
scripts/03_moving_average_baseline.py
scripts/04_single_regime_baseline.py
scripts/05_fit_hmm.py
scripts/06_bayesian_belief.py
scripts/07_bayesian_strategy.py
scripts/08_strategy_comparison.py
scripts/09_regime_and_calibration.py
scripts/10_ablation_studies.py
scripts/11_pgmpy_dbn.py
scripts/11a_pgmpy_exploration.py
scripts/12_all_plots.py
scripts/13_enhanced_agent.py
scripts/run_all.py
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
results/test_beliefs.csv
results/bayesian_strategy_summary.csv
results/bayesian_strategy_daily.csv
results/strategy_comparison.csv
results/equity_curves.png
results/belief_probabilities.png
```

## Notes

Start here for the team-facing explanation:

```text
notes/workflow_overview.md
```
