Ethereum Price Prediction using Cellular Learning Automata.

The module will be taught at GISMA Business School in its AI studio.The module will be delivered in GISMA Business School's AI studio.
The assessment for this module is an Individual Project.There is a single assessment for this module which is an Individual Project due on 3 July 2026.

---

## Overview

This project uses Cellular Learning Automata (CLA) based method to predict the closing price of the Ethereum cryptocurrency (ETH/USD) on a daily basis. Each of the seven Learning Automata (LAs) is associated with one lag of the price series and learns optimal weighting schemes using the Linear Reward-Inaction (LRI) reinforcement rule. The model is trained on the year 2024 and tested on the first 30 days of January 2025.

No external Machine Learning libraries (sklearn, PyTorch, TensorFlow) are used. The whole implementation is done in a single Colab notebok, download data and model, validate, forecast and plot.

---

## Run in Google Colab

Run all cells sequentially in order on top to bottom in Google Colab when you open the file eth_ca_forecast.ipynb. No installation needed locally.

The following dependencies are automatically installed in Cell 1:
```
yfinance  pandas  numpy  matplotlib  seaborn  requests
```

---

## How It Works

### Cellular Learning Automata

An agent that performs reinforcement learning is called a Learning Automaton (LA) and:
2. Initiates an action from a set of actions
2. Is rewarded by the environment
3. Updates its action probability distribution to aid in future decisions

A group of LAs is connected in a structured array, called a **Cellular Learning Automaton**. In this project, each cell represents one lag of the ETH price series in history.

```
Day t-1   Day t-2   Day t-3  ...  Day t-7
   │          │         │             │
 Cell 1    Cell 2    Cell 3  ...   Cell 7
   LA         LA        LA            LA
   │          │         │             │
   └──────────┴─────────┴─────────────┘
                       ↓
           prediction(t) = weighted average of lags
```

Each cell learns a weight (0%, 25%, 50%, 75%, or 100%) for its lag. The forecast is the average (weighted) of all seven lagged values. The weights are updated according to the LRI rule:

```
p_a(t+1) = p_a(t) + α · r · (1 − p_a(t))      ← chosen action
p_j(t+1) = p_j(t) − α · r · p_j(t)             ← all other actions
```

where r = max(0, 1 − |error| / actual) is a reward function that maps to the accuracy of the prediction on a scale from 0 to 1.

### Why 7 cells?

One trading week consists of 7 cells = 7 days. The weekly time frame is the characteristic time frame of crypto markets. At the end of full-year training, the trained weight matrix confirmed this: Cell 7 (lag t−7) had the highest weight (1.0) and Cell 3 (lag t−3) had the lowest weight (0.0).

---

## Notebook Structure

| Section | Description |
|---|---|
1 — Configuration: Hyperparameters, date ranges, random seed |
2 — Download data | ETH/USD daily prices | using the library yfinance |
Graphs of price and return distribution and summary statistics |
4 — Extract series: `close_usd` as a 1-D NumPy array |
This is the fifth LA class showing how to use the choose() and update_lri() functions.
CLA class that fits and predicts next actions as a probability matrix, named CLAForecaster.
Validation: 30-day walk forward on Dec 2024, vs. naive baseline |
8 years of training = 366 retraining days |
Forecast for 9-30 days ahead of the forecast period | Walk-forward forecast for Jan 2025 (vs actuals)
Plots | Price history + forecast vs actual | 10 —
12 — Appendix | Appendix A: Media research conducted by the Aljamiado College of Theology |

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
Number of learning automata / lags | `N_CELLS`
| `N_ACTIONS` | 5 | Weight levels: 0.0, 0.25, 0.50, 0.75, 1.0 |
| `LA_LEARNING_RATE` | 0.08 | LRI update step size |
5 | Training passes over entire series |
The hold-out window for validation.Determines the number of days that will be held back for validation.

---

## Results

### 30-Day Walk-Forward Validation (Dec 2024)

| Model | MAE |
|---|---|
| CLA Forecaster | $144.56 |
| Naive baseline (persistence) | $90.16 |

On the validation window, the CLA doesn't outperform the naive baseline. This should be expected: Crypto prices have high autocorrelation – persistence is a good baseline. Errors are limited and fixed.

### 30-Day Out-of-Sample Forecast (Jan 2025)

| Metric | Value |
|---|---|
| Mean Absolute Error | $120.78 |
| Mean % Error | 3.60% |
| Best single-day error | 0.06% (20 Jan) |
| Worst single-day error | 9.47% (9 Jan) |

The walk-forward forecast incorporates actual prices into the historical record at each iteration, unlike the recursive forecast which is based on convergence at a fixed point.

### Learned Weight Matrix

Following full year training the automata brought together to deterministic weights:

| Cell | Lag | Weight |
|---|---|---|
| 1 | t−1 | 0.50 |
| 2 | t−2 | 0.50 |
| 3 | t−3 | 0.00 — discarded |
| 4 | t−4 | 0.75 |
| 5 | t−5 | 0.50 |
| 6 | t−6 | 0.25 |
| 7 | t−7 | 1.00 — maximum |

---

## Repository Structure

```
eth-ca-forecast/
├── eth_ca_forecast.ipynb   # Main notebook — submit this
├── README.md               # This file
```

---

## References

- Narendra, K.S. and Thathachar, M.A.L. (1989) Learning Automata: An Introduction. Prentice-Hall.
Reinforcement Learning: An Introduction by Sutton, R.S. and Barto, A.G. (2018) 2nd edn. MIT Press.
- Aroussi, R. (2019) *yfinance*. https://github.com/ranaroussi/yfinance

---

## AI Disclosure

Developed with AI assistance (Claude, Anthropic). The professor for the AI Studio module explicitly allows for using AI in this assessment.
