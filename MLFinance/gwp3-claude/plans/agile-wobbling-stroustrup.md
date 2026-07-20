# GWP3 Pipeline Execution Plan

## Context
The GWP3 deep learning script for SPY time series forecasting has been fully implemented but requires Python dependencies to be installed before it can run.

## Dependencies Required
The script imports the following Python packages:
- `numpy` - numerical computing
- `pandas` - data manipulation
- `yfinance` - Yahoo Finance data download (required)
- `matplotlib`
- `seaborn`
- `scikit-learn` - ML utilities and metrics
- `tensorflow` - deep learning (MLP, LSTM)
- `keras` - neural network building blocks

## Implementation Plan

### Step 1: Install Required Packages
Install all dependencies via pip:
```bash
pip install yfinance numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### Step 2: Run the Pipeline
```bash
cd /Users/alberto/Documents/projects/GWP_1/MLFinance/gwp3-claude/code
python gwp3.py
```

### Step 3: Verify Output
Expected outputs:
- Data file: `spy_data.pkl`
- Visualizations: Multiple PNG files (price series, returns distribution, GAF images, accuracy comparisons)
- Console output with analysis results and recommendations

## Files to Be Created/Modified
- **Read-only**: `/Users/alberto/Documents/projects/GWP_1/MLFinance/gwp3-claude/code/gwp3.py` (entry point)
- **Created**: Dependencies will be installed into the Python environment
- **Generated outputs** (after running):
  - `spy_data.pkl`
  - `comparison_accuracy_step1.png`
  - `comparison_walkforward_splits.png`
  - `walkforward_accuracy_time.png`
  - `walkforward_pnl_time.png`
  - `comparison_sharpe_all.png`
  - `gaf_examples.png`
