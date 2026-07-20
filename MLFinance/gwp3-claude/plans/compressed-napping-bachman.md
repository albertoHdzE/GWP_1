# GWP3 Code Fix Plan

## Context
The gwp3.py script has multiple issues that need to be fixed to run successfully.

## Issues Fixed
1. **run_step1 function signature mismatch**: Changed from passing individual arguments to passing a proper data dictionary
2. **backtest_classification loop index error**: Fixed the loop to properly iterate over predictions

## Remaining Issues
The backtest function is receiving an empty `actual_returns` array (size 0). This happens because:
- `predictions` has shape (n_samples, 1) from model.predict()
- `actual_returns` is being sliced incorrectly from y_reg

## Solution

### 1. Fix run_step1 calls in main()
The `run_step1` function expects a `data` dictionary with keys: 'X', 'y_class', 'y_reg'

**File:** `code/gwp3.py`
**Lines:** 1333-1338 and 1408-1413

**For NO LEAKAGE case:**
```python
data_no_leakage = {
    'X': np.concatenate([X_train, X_test]),
    'y_class': np.concatenate([y_class_train, y_class_test]),
    'y_reg': y_reg[train_size:train_size+len(X_test)]
}
results_no_leakage = run_step1(models_no_leakage, data_no_leakage)
```

**For LEAKAGE case:**
```python
data_leakage = {
    'X': np.concatenate([X_train_l, X_test_l]),
    'y_class': np.concatenate([y_class_train_l, y_class_test_l]),
    'y_reg': y_reg_l[train_size_l:train_size_l+len(X_test_l)]
}
results_leakage = run_step1(models_leakage, data_leakage)
```

### 2. Fix backtest_classification loop
**File:** `code/gwp3.py`
**Lines:** 814-816

Change from:
```python
for i in range(1, n):
    if positions[i-1] == 1:
        strategy_returns[i] = actual_returns[i]
```

To:
```python
for i in range(n-1):
    if positions[i] == 1:
        strategy_returns[i+1] = actual_returns[i]
```

### 3. Ensure y_reg has correct length in main()
The y_reg array passed to run_step1 must have exactly `n_train + n_test` elements.

## Verification
After making these changes, run:
```bash
python3 code/gwp3.py
```

The script should complete all steps and generate all required plots in the `report/images/` directory.
