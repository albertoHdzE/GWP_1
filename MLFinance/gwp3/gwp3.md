# MScFE 642 Deep Learning for Finance — Group Work Project \#3 (GWP3)

## Accurate project prompt (copy/paste)

You are completing MScFE 642 Deep Learning for Finance — Group Work Project \#3 as a team of three (but you will implement everything end-to-end). Choose a single financial security (equity, cryptocurrency, options, bonds, volatility index, yields, etc.) and use a time series of **no more than 2,000 observations**.

**Step 1**

1. **(1a)** Gather information on the security’s price time series. Include a clear graphical and textual description of the dataset used.
2. **(1b)** Build a predictive model to forecast the security’s **returns** (or volatilities, yield changes, etc.). Construct the target labels in a way that makes it apparent that **information leakage exists** between the training and test samples in the predictive modeling setup.
3. **(1c)** Using a **single train/test split**, train and evaluate **three deep learning models** to predict the time series:
   - (i) an **MLP**
   - (ii) an **LSTM**
   - (iii) a **CNN based on GAF** (Gramian Angular Field)
4. **(1d)** Provide results from **backtests** of the trading strategies that arise from each of the three models.

**Step 2 (walk-forward backtests, leakage present)**

Backtest the models with a **non-anchored walk-forward** method:

1. **(2a)** Use a train/test split with **500 observations in each set** (500 train, 500 test) per walk-forward iteration.
2. **(2b)** Use a train/test split with **500 observations in each training set** and **100 observations in the test sets** per iteration.
3. **(2c)** Discuss how the backtest results in (2a) and (2b) change compared to Step 1.
4. **(2d)** Compare (2a) vs (2b). Can backtest overfitting due to leakage explain the results?

**Step 3 (walk-forward backtests, leakage reduced)**

Repeat Step 2, but **alleviate leakage** between training and test sets:

1. **(3a)** Propose, set up, and describe a method that reduces leakage between training and test samples.
2. **(3b)** Using that method, run non-anchored walk-forward with **500/500** train/test.
3. **(3c)** Using that method, run non-anchored walk-forward with **500/100** train/test.
4. **(3d)** Compare (3b) vs (3c). Has overfitting apparently disappeared relative to Step 2/Step 1?

**Step 4 (submission organization)**

Organize:

- One executable Jupyter notebook with answers for all steps; ensure each section begins with the **question number** (e.g., “Step 3a. …”).
- A report (PDF) containing **only answers** (no code) in a clear, professional style.

Submission requirements (as stated): one team member submits **(i)** a PDF with answers only and **(ii)** a zip file containing the executable notebook and an HTML export of the notebook.

## Deliverables in this repository

- `gwp3.md` (this file): prompt + step-by-step plan for execution.
- `report/report.tex`: LaTeX source for the “answers only” report (compile to PDF).
- One Jupyter notebook (to be created separately) containing code + outputs + plots, organized by question number.

## Team-of-3 execution plan (we complete all tasks)

### Shared project decisions (do first)

1. Pick the security and frequency (daily is simplest). Hard constraint: keep the final series ≤ 2,000 observations.
2. Fix the prediction target:
   - Regression: next-day return, next-week return, next-day realized volatility, yield change, etc.
   - Classification: direction (up/down), quantile buckets, volatility regime, etc.
3. Fix the feature set (start simple, then add):
   - Returns, rolling stats (mean/std), momentum, RSI, MACD, rolling volatility, volume features (if available).
4. Fix evaluation and backtest conventions (must be consistent across steps):
   - Prediction horizon, signal mapping from prediction → position, transaction costs, position constraints.

### Step 1 plan (single split, leakage intentionally present)

**1a. Data + EDA**

- Download and clean prices.
- Create the working time series (close-to-close returns or relevant transform).
- Provide plots: price series, return series, histogram, autocorrelation (optional), rolling volatility.
- Summarize dataset: date range, number of observations, missing data handling.

**1b. Intentionally introduce leakage (make it explicit)**

Implement a setup where it is obvious that future information contaminates training/test, for example:

- Use a scaler (standardization/normalization) fit on the **full dataset** before splitting.
- Use feature engineering that inadvertently uses future values (e.g., centered rolling windows, forward-filled information not available at trade time, computing labels using full-sample statistics).
- Perform label construction and/or feature transformation globally, then split, without respecting time order.

Document the leakage mechanism explicitly in the notebook and report (what leaked, why it is leakage, and why it makes results look too good).

**1c. Train three DL models (single split)**

- Define a single train/test split (chronological).
- Create windowed sequences for MLP/LSTM (e.g., lookback length L).
- For CNN+GAF:
  - Convert each lookback window into a Gramian Angular Field image.
  - Train a CNN to predict the label from GAF images.
- Track:
  - Train/validation curves, hyperparameters, and final test metrics.
  - Baseline comparator (e.g., naive “predict 0 return” for regression or “predict majority class” for classification).

**1d. Backtest strategies from each model**

- Convert model output to trading signals:
  - Regression: long if prediction > 0, short if prediction < 0 (optionally add threshold).
  - Classification: long if prob(up) > 0.5, short otherwise (or use thresholds).
- Backtest on the test segment of the single split.
- Report metrics and plots:
  - Equity curve, drawdown, turnover.
  - CAGR/annualized return, volatility, Sharpe, max drawdown, hit rate, average trade return.
- Make sure plots include axes labels/scales.

### Step 2 plan (walk-forward, leakage still present)

Non-anchored walk-forward means each iteration trains on a rolling window and tests on the following window; the training window moves forward each time (not expanding).

**2a. 500/500 rolling**

- For each fold: train on 500 observations, test on the next 500.
- Aggregate fold results into a single backtest (concatenate test periods).
- Compare metrics to Step 1.

**2b. 500/100 rolling**

- For each fold: train on 500 observations, test on the next 100.
- Aggregate fold results; compare metrics to Step 1 and to 2a.

**2c–2d. Discussion**

- Explain why shorter test windows often look better when leakage/overfitting is present.
- Tie observed performance changes to leakage and model instability across folds.

### Step 3 plan (walk-forward, leakage reduced)

**3a. Leakage-reduction method (choose one, make it reproducible)**

Use a time-series-safe pipeline, such as:

- Fit scalers/normalizers **inside each fold, using only the training window**, then transform train and test separately.
- Ensure every feature is computable at time t using information available at or before t (use trailing windows only; avoid centered windows).
- If labeling uses future returns (it must), ensure that the label aligns to the decision time and that features do not include the label horizon.
- Use purging/embargo around split boundaries if overlapping information exists due to lookback/horizon:
  - Purge: drop samples whose label window overlaps the test window.
  - Embargo: skip a small buffer after training before testing.

Clearly state the final chosen method and why it reduces leakage.

**3b–3c. Re-run walk-forward**

- Repeat 500/500 and 500/100 with the leakage-reduced pipeline.
- Keep everything else the same (signal mapping, costs, metrics) for fair comparison.

**3d. Discussion**

- Contrast Step 2 vs Step 3; identify whether “too good to be true” performance disappears.
- Discuss practical implications for deploying such a model.

### Step 4 plan (final packaging)

- Notebook:
  - One notebook containing all code, outputs, and figures.
  - Every section starts with the question label (e.g., “Step 2b. …”).
  - Avoid references to code function names in the report.
- Report (answers only):
  - Use `report/report.tex` as the starting template.
  - Include clear narrative, professional formatting, labeled figures/tables, and a bibliography in MLA format.

## Work allocation across 3 members (internal)

- Member A (Data & leakage design): Step 1a–1b, dataset description, leakage demonstration.
- Member B (Models): Step 1c model training (MLP, LSTM, CNN+GAF), hyperparameter logging, metrics.
- Member C (Backtesting & reporting): Step 1d, Step 2 and Step 3 walk-forward backtests, performance analysis, LaTeX report integration.

All members review Step 2–3 discussion for consistency and ensure final report matches the notebook outputs.
