# Workflow Overview

This folder contains the algorithm side of the stock-trading project. The goal
is to keep the implementation simple, documented, and easy to explain.

## 1. Current Status

Completed so far:

```text
data loading
data sanity checks
real-return calculation
three planned baselines
basic HMM parameter estimation with hmmlearn
basic Bayesian belief updates
basic belief-based portfolio strategy
strategy comparison table
simple result plots
```

Still pending:

```text
pgmpy exploration / Bayesian network connection
validation/model selection
strategy improvement decisions
```

## 2. Current Workflow

```text
S&P 500 CSV files
    |
    v
load train/test data
    |
    v
check data ranges, columns, missing values
    |
    v
compute real log returns from Adj_Close
    |
    v
run three baseline strategies
    |
    v
fit a K=3 Gaussian HMM on training features
    |
    v
run Bayesian belief updates on test observations
    |
    v
convert beliefs into daily portfolio weights
    |
    v
backtest the belief-based strategy
    |
    v
compare all strategies
    |
    v
plot equity curves and belief probabilities
    |
    v
save summaries in results/
```

## 3. Data Stage

Data files:

```text
data/sp500_train.csv
data/sp500_test.csv
```

Main loader:

```text
src/data.py
```

The loader:

```text
reads CSV files
parses Date
sorts by Date
sets Date as the index
returns train/test DataFrames
```

The model feature columns are:

```text
Log_Return
Volume
Volatility
```

We also compute:

```text
Real_Log_Return = log(Adj_Close_t / Adj_Close_{t-1})
```

`Real_Log_Return` is used for performance/backtesting because it comes from
actual price movement.

## 4. Data Summary Script

Script:

```text
scripts/01_data_summary.py
```

Purpose:

```text
print train/test date ranges
print row counts
print column names
check missing values
show feature statistics
show real-return statistics
```

Current data checks:

```text
training range: 2015-01-02 to 2020-12-31
training rows: 1511
test range: 2021-01-04 to 2025-12-30
test rows: 1254
missing values: none
```

## 5. Baselines

The original project plan asks for three baselines:

```text
Buy and Hold
Moving Average Crossover
Agent without explicit belief states
```

In this pipeline, these are implemented as:

```text
Buy and Hold
Moving Average Crossover 50/200
Single-Regime No-Belief
```

Baseline helper:

```text
src/baselines.py
```

Backtesting helper:

```text
src/backtest.py
```

## 6. Baseline Scripts

Buy and Hold:

```text
scripts/02_buy_hold_baseline.py
results/buy_hold_summary.csv
```

Moving Average Crossover:

```text
scripts/03_moving_average_baseline.py
results/moving_average_summary.csv
```

Single-Regime No-Belief:

```text
scripts/04_single_regime_baseline.py
results/single_regime_summary.csv
```

Current baseline test results:

| Strategy | Total Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Buy and Hold | 85.04% | 0.739 | -25.43% |
| Moving Average Crossover 50/200 | 33.86% | 0.486 | -19.74% |
| Single-Regime No-Belief | 55.12% | 0.739 | -18.88% |

## 7. HMM Parameter Estimation

This completes the first algorithm task from the project note:

```text
Implement HMM parameter estimation (hmmlearn)
```

Files:

```text
src/hmm_model.py
scripts/05_fit_hmm.py
```

The current HMM setup:

```text
library: hmmlearn
model: GaussianHMM
hidden states: K = 3
features: Log_Return, Volume, Volatility
training data only
```

Current HMM outputs:

```text
results/hmm_model.pkl
results/hmm_state_summary.csv
results/hmm_transition_matrix.csv
results/hmm_training_summary.txt
```

Current HMM result summary:

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

## 8. Bayesian Belief Updates

The second algorithm task from the project note is implemented:

```text
Implement Bayesian belief updates
```

Files:

```text
src/belief_update.py
scripts/06_bayesian_belief_update.py
```

Output:

```text
results/test_beliefs.csv
```

The belief update uses:

```text
HMM transition matrix
Gaussian observation likelihoods
test-set observations
```

to produce daily probabilities for each hidden state.

## 9. Belief-Based Portfolio Strategy

The basic Bayesian strategy is implemented.

Files:

```text
src/portfolio.py
scripts/07_bayesian_strategy.py
```

Outputs:

```text
results/bayesian_strategy_summary.csv
results/bayesian_strategy_daily.csv
```

Current test-period result:

| Strategy | Total Return | Sharpe | Max Drawdown | Average Weight |
|---|---:|---:|---:|---:|
| Bayesian Belief-State | 31.79% | 0.615 | -14.25% | 0.611 |

## 10. What Has Not Been Done Yet

The end-to-end skeleton is running, but the Bayesian strategy has not been
improved or validated yet.

Completed skeleton outputs:

```text
results/strategy_comparison.csv
results/equity_curves.png
results/belief_probabilities.png
```

Remaining project work:

```text
explore pgmpy for Bayesian network/DBN representation
add validation/model selection
decide how to improve the belief update and portfolio rule
```

## 11. How To Run Current Scripts

Run these from the `belief_state_trader` folder.

Full end-to-end run:

```text
python scripts/run_all.py
```

Data summary:

```text
python scripts/01_data_summary.py
```

Baselines:

```text
python scripts/02_buy_hold_baseline.py
python scripts/03_moving_average_baseline.py
python scripts/04_single_regime_baseline.py
```

HMM fitting:

```text
python scripts/05_fit_hmm.py
```

Bayesian belief update and strategy:

```text
python scripts/06_bayesian_belief_update.py
python scripts/07_bayesian_strategy.py
```

Comparison and plots:

```text
python scripts/08_strategy_comparison.py
python scripts/09_make_plots.py
```

## 12. Short Team Update

Basic HMM parameter estimation is implemented and running in the clean
pipeline. The current HMM uses `K = 3` hidden states and the features
`Log_Return`, `Volume`, and `Volatility`. The clean pipeline also includes
data loading, data checks, real-return calculation, the three planned
baselines, basic Bayesian belief updates, and a first belief-based strategy.
The skeleton also creates a combined strategy comparison table and basic plots
for equity curves and belief probabilities.
