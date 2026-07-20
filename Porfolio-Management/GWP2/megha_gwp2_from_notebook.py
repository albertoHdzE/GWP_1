
# ===== Cell 0: code =====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import cvxpy as cp
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Plotting style
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_theme(style='whitegrid')

print('All libraries imported successfully.')



# ===== Cell 1: markdown =====

# # 1. Asset selection


# ===== Cell 2: code =====

# ── Asset universe ──────────────────────────────────────────────────────────
TICKERS = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM',
           'AMZN', 'PG',   'NEE', 'GLD', 'TLT', 'VNQ', 'BRK-B']

# ── Date ranges ─────────────────────────────────────────────────────────────
TRAIN_START = '2024-01-01'
TRAIN_END   = '2025-09-30'
TEST_START  = '2025-10-01'
TEST_END    = '2025-12-31'

TRADING_DAYS = 252   # annualisation constant
RF_ANNUAL    = 0.045 # risk-free rate (approx. Fed Funds 2024-2025)
RF_DAILY     = RF_ANNUAL / TRADING_DAYS

print(f'Portfolio: {len(TICKERS)} assets')
print(f'Training : {TRAIN_START} → {TRAIN_END}')
print(f'Testing  : {TEST_START}  → {TEST_END}')



# ===== Cell 3: markdown =====

# # 2. Downloading and preprocessing


# ===== Cell 4: code =====

# Download adjusted close prices for the full period
raw = yf.download(
    TICKERS,
    start=TRAIN_START,
    end='2026-01-01',   # slightly beyond TEST_END to capture Dec 31
    auto_adjust=True,
    progress=False
)['Close']

# Drop any tickers that failed to download
raw = raw.dropna(axis=1, how='all')
raw = raw.ffill().dropna()

print(f'Downloaded {raw.shape[1]} assets × {raw.shape[0]} trading days')
print(f'Date range in data: {raw.index[0].date()} → {raw.index[-1].date()}')
raw.tail(3)



# ===== Cell 5: code =====

# Daily log returns
returns_all = np.log(raw / raw.shift(1)).dropna()

# Split into in-sample and out-of-sample
returns_train = returns_all.loc[TRAIN_START:TRAIN_END]
returns_test  = returns_all.loc[TEST_START:TEST_END]

print(f'Training rows : {len(returns_train)} trading days')
print(f'Testing rows  : {len(returns_test)}  trading days')



# ===== Cell 6: code =====

# ── Summary statistics for the training period ───────────────────────────────
ann_ret  = returns_train.mean() * TRADING_DAYS
ann_vol  = returns_train.std()  * np.sqrt(TRADING_DAYS)
sharpe   = (ann_ret - RF_ANNUAL) / ann_vol

summary = pd.DataFrame({
    'Ann. Return': ann_ret,
    'Ann. Volatility': ann_vol,
    'Sharpe Ratio': sharpe
}).round(4)

print('=== Training-Period Asset Summary ===')
display(summary)



# ===== Cell 7: code =====

# ── Correlation heatmap ───────────────────────────────────────────────────────
corr = returns_train.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax)
ax.set_title('Asset Correlation Matrix – Training Period (2024-01-01 to 2025-09-30)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print('Figure saved: correlation_heatmap.png')



# ===== Cell 8: markdown =====

# # 3. Step 1: Optimal Kelly Porflio


# ===== Cell 9: code =====

# ── In-sample parameters ─────────────────────────────────────────────────────
mu    = returns_train.mean().values          # daily mean log-returns
Sigma = returns_train.cov().values           # daily covariance matrix
n     = len(mu)

mu_excess = mu - RF_DAILY                    # excess returns over risk-free

print(f'Number of assets : {n}')
print(f'Daily μ (first 3): {mu[:3].round(6)}')



# ===== Cell 10: code =====

# ── 3.2 Unconstrained Kelly (analytical) ─────────────────────────────────────
Sigma_inv = np.linalg.inv(Sigma)
w_kelly_raw = Sigma_inv @ mu_excess

# Normalise so weights sum to 1 (fully invested)
w_kelly_unconstrained = w_kelly_raw / w_kelly_raw.sum()

kelly_unc_df = pd.Series(
    w_kelly_unconstrained,
    index=returns_train.columns,
    name='Unconstrained Kelly Weight'
).round(4)

print('=== Unconstrained Kelly Weights ===')
print(kelly_unc_df.to_string())
print(f'\nSum of weights: {kelly_unc_df.sum():.4f}')
print(f'Max weight    : {kelly_unc_df.max():.4f}')
print(f'Min weight    : {kelly_unc_df.min():.4f}')



# ===== Cell 11: code =====

# ── 3.3 Constrained Kelly (cvxpy) ────────────────────────────────────────────
# Constraints:
#   1. weights sum to 1  (fully invested)
#   2. 0 <= w_i <= 0.20  (no short selling; max 20% per asset)

w = cp.Variable(n)

kelly_objective = cp.Maximize(
    mu_excess @ w - 0.5 * cp.quad_form(w, Sigma)
)

kelly_constraints = [
    cp.sum(w) == 1,
    w >= 0,
    w <= 0.20
]

prob_kelly = cp.Problem(kelly_objective, kelly_constraints)
prob_kelly.solve(solver=cp.CLARABEL)

if prob_kelly.status not in ['optimal', 'optimal_inaccurate']:
    raise ValueError(f'Kelly solver failed: {prob_kelly.status}')

w_kelly = np.array(w.value)
w_kelly = np.clip(w_kelly, 0, None)          # numerical clean-up
w_kelly /= w_kelly.sum()                      # re-normalise

kelly_df = pd.Series(
    w_kelly,
    index=returns_train.columns,
    name='Constrained Kelly Weight'
).round(4)

print('=== Constrained Kelly Weights (max 20% per asset) ===')
print(kelly_df.to_string())
print(f'\nSum of weights: {kelly_df.sum():.4f}')
print(f'Max weight    : {kelly_df.max():.4f}')



# ===== Cell 12: code =====

# ── Double-Kelly weights (half Kelly) ────────────────────────────────────────
# Double-Kelly = 2 × Kelly (leveraged); Half-Kelly = 0.5 × Kelly
# Per the assignment, we compare Kelly vs Double-Kelly
# Double-Kelly weights are 2x the Kelly weights (may exceed 1 -> leveraged)
w_double_kelly = 2.0 * w_kelly

dk_df = pd.Series(
    w_double_kelly,
    index=returns_train.columns,
    name='Double-Kelly Weight'
).round(4)

# ── Half-Kelly weights ───────────────────────────────────────────────────────
w_half_kelly = 0.5 * w_kelly

hk_df = pd.Series(
    w_half_kelly,
    index=returns_train.columns,
    name='Half-Kelly Weight'
).round(4)

# ── Comparison table ─────────────────────────────────────────────────────────
weights_comparison = pd.DataFrame({
    'Unconstrained Kelly': kelly_unc_df,
    'Kelly (≤20%)': kelly_df,
    'Double-Kelly': dk_df,
    'Half-Kelly': hk_df
})

print('=== Weight Comparison Table ===')
display(weights_comparison.round(4))
print('\nColumn sums:')
print(weights_comparison.sum().round(4))



# ===== Cell 13: code =====

# ── Weight bar chart ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Constrained Kelly
kelly_df.sort_values(ascending=True).plot(
    kind='barh', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].axvline(0.20, color='red', linestyle='--', label='20% cap')
axes[0].set_title('Constrained Kelly Portfolio Weights', fontweight='bold')
axes[0].set_xlabel('Weight')
axes[0].legend()

# Side-by-side Kelly vs Double-Kelly
x = np.arange(len(TICKERS))
width = 0.35
axes[1].bar(x - width/2, w_kelly,        width, label='Kelly',        color='steelblue', edgecolor='black')
axes[1].bar(x + width/2, w_double_kelly, width, label='Double-Kelly',  color='darkorange', edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(returns_train.columns, rotation=45, ha='right')
axes[1].set_title('Kelly vs Double-Kelly Weights', fontweight='bold')
axes[1].set_ylabel('Weight')
axes[1].legend()

plt.tight_layout()
plt.savefig('kelly_weights.png', dpi=150)
plt.show()
print('Figure saved: kelly_weights.png')



# ===== Cell 14: markdown =====

# # Step 2: Performance comparison: comparing 3 strategies on period Oct 1 to Dec 31, 2025


# ===== Cell 15: code =====

# ── Portfolio daily returns (out-of-sample) ───────────────────────────────────
def portfolio_returns(weights, ret_df):
    """Compute daily portfolio return series given weight vector."""
    return ret_df.values @ weights   # shape: (T,)

r_kelly        = portfolio_returns(w_kelly,        returns_test)
r_double_kelly = portfolio_returns(w_double_kelly, returns_test)
r_half_kelly   = portfolio_returns(w_half_kelly,   returns_test)

port_returns = pd.DataFrame({
    'Kelly': r_kelly,
    'Double-Kelly': r_double_kelly,
    'Half-Kelly': r_half_kelly
}, index=returns_test.index)

print('Out-of-sample daily return statistics:')
display(port_returns.describe().round(6))



# ===== Cell 16: code =====

# ── Metric helper functions ────────────────────────────────────────────────────

def cumulative_return(r):
    """Total cumulative return over the period."""
    return np.exp(np.sum(r)) - 1

def annualised_sharpe(r, rf_daily=RF_DAILY, trading_days=TRADING_DAYS):
    """Annualised Sharpe ratio (log-return version)."""
    excess = r - rf_daily
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(trading_days)

def max_drawdown(r):
    """Maximum Drawdown from cumulative log-return series."""
    cum = np.exp(np.cumsum(r))
    rolling_max = np.maximum.accumulate(cum)
    drawdowns = (cum - rolling_max) / rolling_max
    return drawdowns.min()   # negative value

def cvar_95(r):
    """Historical 95%-CVaR (Expected Shortfall)."""
    var_05 = np.percentile(r, 5)
    return r[r <= var_05].mean()   # negative value



# ===== Cell 17: code =====

# ── Compute metrics for each strategy ────────────────────────────────────────
strategies = {
    'Kelly': r_kelly,
    'Double-Kelly': r_double_kelly,
    'Half-Kelly': r_half_kelly
}

rows = []
for name, r in strategies.items():
    rows.append({
        'Strategy': name,
        'Cumulative Return': f"{cumulative_return(r):.4%}",
        'Sharpe Ratio':      f"{annualised_sharpe(r):.4f}",
        'Max Drawdown':      f"{max_drawdown(r):.4%}",
        '95%-CVaR (daily)':  f"{cvar_95(r):.4%}"
    })

results_table = pd.DataFrame(rows).set_index('Strategy')

print('='*65)
print('  OUT-OF-SAMPLE PERFORMANCE (Oct 1 – Dec 31, 2025)')
print('='*65)
display(results_table)
print('='*65)
print('Note: CVaR and Max Drawdown are negative (losses).')



# ===== Cell 18: code =====

# ── Numeric version for charting ──────────────────────────────────────────────
rows_num = []
for name, r in strategies.items():
    rows_num.append({
        'Strategy': name,
        'Cumulative Return': cumulative_return(r),
        'Sharpe Ratio':      annualised_sharpe(r),
        'Max Drawdown':      max_drawdown(r),
        '95%-CVaR (daily)':  cvar_95(r)
    })

results_num = pd.DataFrame(rows_num).set_index('Strategy')
print('Numeric metrics computed.')
display(results_num.round(4))



# ===== Cell 19: code =====

# ── Cumulative return chart ───────────────────────────────────────────────────
cum_returns = np.exp(port_returns.cumsum()) - 1   # convert log-returns to simple cumulative

fig, ax = plt.subplots(figsize=(12, 5))
colors = {'Kelly': 'steelblue', 'Double-Kelly': 'darkorange', 'Half-Kelly': 'forestgreen'}
for col in cum_returns.columns:
    ax.plot(cum_returns.index, cum_returns[col] * 100,
            label=col, linewidth=2, color=colors[col])

ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.set_title('Cumulative Returns – Out-of-Sample Period (Oct–Dec 2025)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Return (%)')
ax.set_xlabel('Date')
ax.legend()
plt.tight_layout()
plt.savefig('cumulative_returns_oos.png', dpi=150)
plt.show()
print('Figure saved: cumulative_returns_oos.png')



# ===== Cell 20: code =====

# ── Drawdown chart ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

for col, r in strategies.items():
    cum = np.exp(np.cumsum(r))
    rolling_max = np.maximum.accumulate(cum)
    dd = (cum - rolling_max) / rolling_max * 100
    ax.plot(returns_test.index, dd, label=col, linewidth=2, color=colors[col])

ax.fill_between(returns_test.index,
                (np.exp(np.cumsum(r_double_kelly)) -
                 np.maximum.accumulate(np.exp(np.cumsum(r_double_kelly)))) /
                np.maximum.accumulate(np.exp(np.cumsum(r_double_kelly))) * 100,
                0, alpha=0.1, color='darkorange')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.set_title('Portfolio Drawdown – Out-of-Sample Period (Oct–Dec 2025)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.legend()
plt.tight_layout()
plt.savefig('drawdown_oos.png', dpi=150)
plt.show()
print('Figure saved: drawdown_oos.png')



# ===== Cell 21: code =====

# ── Metrics bar chart ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

metrics = ['Cumulative Return', 'Sharpe Ratio', 'Max Drawdown', '95%-CVaR (daily)']
bar_colors = [colors[s] for s in results_num.index]

for ax, metric in zip(axes.flatten(), metrics):
    vals = results_num[metric]
    bars = ax.bar(vals.index, vals.values * 100 if 'Return' in metric or 'Drawdown' in metric or 'CVaR' in metric
                  else vals.values,
                  color=bar_colors, edgecolor='black', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(metric, fontweight='bold')
    if 'Return' in metric or 'Drawdown' in metric or 'CVaR' in metric:
        ax.set_ylabel('(%)')
    else:
        ax.set_ylabel('Ratio')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h,
                f'{h:.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=10)

plt.suptitle('Out-of-Sample Risk & Performance Metrics (Oct–Dec 2025)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure saved: metrics_comparison.png')



# ===== Cell 22: code =====

# ── Return distribution (daily) ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (name, r) in zip(axes, strategies.items()):
    ax.hist(r * 100, bins=25, color=colors[name], edgecolor='black', alpha=0.8, density=True)
    var5 = np.percentile(r, 5) * 100
    ax.axvline(var5, color='red', linestyle='--', label=f'VaR 5% = {var5:.2f}%')
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Daily Return (%)')
    ax.legend(fontsize=9)

axes[0].set_ylabel('Density')
plt.suptitle('Daily Return Distributions – Out-of-Sample Period',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('return_distributions.png', dpi=150)
plt.show()
print('Figure saved: return_distributions.png')



# ===== Cell 23: markdown =====

# # Summary


# ===== Cell 24: code =====

print('\n' + '='*65)
print('  STEP 1: CONSTRAINED KELLY WEIGHTS (≤ 20% per asset)')
print('='*65)
display(kelly_df.to_frame())

print('\n' + '='*65)
print('  STEP 2: OUT-OF-SAMPLE PERFORMANCE COMPARISON')
print('  Period: October 1, 2025 – December 31, 2025')
print('='*65)
display(results_table)

print('\n  Saved figures:')
for f in ['correlation_heatmap.png', 'kelly_weights.png',
          'cumulative_returns_oos.png', 'drawdown_oos.png',
          'metrics_comparison.png', 'return_distributions.png']:
    print(f'    • {f}')



# ===== Cell 25: markdown =====

# # Step 5: ML-Enhanced Portfolio Construction


# ===== Cell 26: markdown =====

# ## A – Covariance Denoising (Marchenko-Pastur / Random Matrix Theory)


# ===== Cell 27: code =====

def denoise_covariance(S, T, N):
    q = T / N                          # concentration ratio

    # Convert to correlation
    std  = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    # Eigendecomposition (ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Marchenko-Pastur upper and lower bounds (sigma^2 = 1 for corr matrix)
    sigma2     = 1.0
    lambda_max = sigma2 * (1 + np.sqrt(1 / q)) ** 2
    lambda_min = sigma2 * (1 - np.sqrt(1 / q)) ** 2

    print(f"  MP noise band       : [{lambda_min:.4f},  {lambda_max:.4f}]")
    noise_mask = (eigenvalues >= lambda_min) & (eigenvalues <= lambda_max)
    print(f"  Noisy eigenvalues   : {noise_mask.sum()} / {N}")

    # Shrink noise eigenvalues to their mean
    eigenvalues_d = eigenvalues.copy()
    if noise_mask.sum() > 0:
        eigenvalues_d[noise_mask] = eigenvalues[noise_mask].mean()

    # Reconstruct
    corr_d = eigenvectors @ np.diag(eigenvalues_d) @ eigenvectors.T
    corr_d = (corr_d + corr_d.T) / 2
    np.fill_diagonal(corr_d, 1.0)
    S_d    = corr_d * np.outer(std, std)
    return S_d



# ===== Cell 28: code =====

T_train, N = returns_train.shape
print("=== Denoising ===")
Sigma_denoised = denoise_covariance(Sigma, T_train, N)
print("Denoised covariance matrix computed.")



# ===== Cell 29: markdown =====

# ## B – Covariance Detoning (Market-Factor Removal)


# ===== Cell 30: code =====

def detone_covariance(S):
    """
    Remove the dominant (market) eigenvector from the covariance matrix.
"""
    std  = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    idx_mkt = np.argmax(eigenvalues)       # largest eigenvalue = market mode
    v_mkt   = eigenvectors[:, idx_mkt]
    pct     = eigenvalues[idx_mkt] / eigenvalues.sum() * 100
    print(f"  Market eigenvalue   : {eigenvalues[idx_mkt]:.4f}  ({pct:.1f}% of trace)")

    corr_d = corr - eigenvalues[idx_mkt] * np.outer(v_mkt, v_mkt)

    # Re-normalise diagonal to 1
    d = np.sqrt(np.diag(corr_d))
    corr_d = corr_d / np.outer(d, d)
    np.fill_diagonal(corr_d, 1.0)

    S_d = corr_d * np.outer(std, std)
    S_d = (S_d + S_d.T) / 2
    return S_d



# ===== Cell 31: code =====

print("=== Detoning ===")
Sigma_detoned = detone_covariance(Sigma)
print("\n=== Denoised → Detoned (stacked) ===")
Sigma_denoised_detoned = detone_covariance(Sigma_denoised)



# ===== Cell 32: markdown =====

# ## C – Hierarchical Risk Parity (HRP)


# ===== Cell 33: code =====

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def hrp_portfolio(returns_df):
    """
    Compute Hierarchical Risk Parity weights following de Prado (2016).
    """
    corr_m = returns_df.corr().values
    cov_m  = returns_df.cov().values
    N_     = corr_m.shape[0]

    # Distance (de Prado 2016)
    dist      = np.sqrt(np.clip((1 - corr_m) / 2, 0, 1))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    link = linkage(condensed, method='single')

    # Quasi-diagonalisation
    def get_quasi_diag(lnk):
        lnk       = lnk.astype(int)
        sort_ix   = pd.Series([lnk[-1, 0], lnk[-1, 1]])
        num_items = lnk[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, len(sort_ix) * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i, j = df0.index, df0.values - num_items
            sort_ix[i]  = lnk[j, 0]
            new_s       = pd.Series(lnk[j, 1], index=i + 1)
            sort_ix     = pd.concat([sort_ix, new_s]).sort_index()
            sort_ix.index = range(len(sort_ix))
        return sort_ix.tolist()

    sort_ix = get_quasi_diag(link)

    def cluster_var(cov, items):
        sub = cov[np.ix_(items, items)]
        w_  = 1.0 / np.diag(sub); w_ /= w_.sum()
        return float(w_ @ sub @ w_)

    def rec_bisect(cov, sx):
        w = pd.Series(1.0, index=sx)
        clusters = [sx]
        while clusters:
            clusters = [i[j:k] for i in clusters
                        for j, k in ((0, len(i)//2), (len(i)//2, len(i)))
                        if len(i) > 1]
            for i in range(0, len(clusters), 2):
                c0, c1 = clusters[i], clusters[i+1]
                v0, v1 = cluster_var(cov, c0), cluster_var(cov, c1)
                alpha  = 1 - v0 / (v0 + v1)
                w[c0] *= alpha; w[c1] *= (1 - alpha)
        return w

    ws = rec_bisect(cov_m, sort_ix).sort_index()
    wa = ws.values.astype(float); wa /= wa.sum()
    return wa



# ===== Cell 34: code =====

print("=== HRP Weights ===")
w_hrp = hrp_portfolio(returns_train)
hrp_df = pd.Series(w_hrp, index=returns_train.columns, name='HRP Weight').round(4)
print(hrp_df.to_string())
print(f"\nSum: {hrp_df.sum():.4f}")



# ===== Cell 35: markdown =====

# ## D – reoptimise Kelly with ML Improved Covariances


# ===== Cell 36: code =====

def solve_kelly(mu_ex, S, max_w=0.20):
    """Constrained Kelly: fully invested, no shorts, max 20% per asset."""
    n_ = len(mu_ex)
    w  = cp.Variable(n_)
    prob = cp.Problem(
        cp.Maximize(mu_ex @ w - 0.5 * cp.quad_form(w, S)),
        [cp.sum(w) == 1, w >= 0, w <= max_w]
    )
    prob.solve(solver=cp.CLARABEL)
    wv = np.clip(np.array(w.value), 0, None)
    return wv / wv.sum()



# ===== Cell 37: code =====

w_denoised_kelly         = solve_kelly(mu_excess, Sigma_denoised)
w_detoned_kelly          = solve_kelly(mu_excess, Sigma_detoned)
w_denoised_detoned_kelly = solve_kelly(mu_excess, Sigma_denoised_detoned)



# ===== Cell 38: code =====

weights_ml = pd.DataFrame({
    'Kelly (baseline)':       w_kelly,
    'Denoised-Kelly':         w_denoised_kelly,
    'Detoned-Kelly':          w_detoned_kelly,
    'Denoised+Detoned-Kelly': w_denoised_detoned_kelly,
    'HRP':                    w_hrp,
}, index=returns_train.columns).round(4)



# ===== Cell 39: code =====

print("=== Step 5 – ML-Enhanced Portfolio Weights ===")
display(weights_ml)



# ===== Cell 40: code =====

# plot it
fig, ax = plt.subplots(figsize=(13, 5))
x  = np.arange(N); bw = 0.15
for i, (label, col) in enumerate(weights_ml.items()):
    ax.bar(x + i * bw, col.values, width=bw, label=label, alpha=0.85)
ax.set_xticks(x + bw * 2)
ax.set_xticklabels(returns_train.columns, rotation=45, ha='right')
ax.set_title('ML-Enhanced Portfolio Weights vs Baseline Kelly', fontweight='bold', fontsize=13)
ax.set_ylabel('Weight'); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('step5_weights_comparison.png', dpi=150)
plt.show()
print('Figure saved: step5_weights_comparison.png')



# ===== Cell 41: markdown =====

# # Step 6: Performance Comparison – All 7 Strategies


# ===== Cell 42: code =====

# performance metrics
def port_returns(w, ret_df):
    return ret_df.values @ w

def cum_ret(r):
    return np.exp(np.sum(r)) - 1

def ann_sharpe(r):
    exc = r - RF_DAILY
    return (exc.mean() / exc.std()) * np.sqrt(TRADING_DAYS) if exc.std() > 0 else np.nan

def sortino(r):
    exc  = r - RF_DAILY
    down = exc[exc < 0]
    dsd  = np.sqrt((down**2).mean()) * np.sqrt(TRADING_DAYS) if len(down) > 0 else np.nan
    return exc.mean() * TRADING_DAYS / dsd if dsd and dsd > 0 else np.nan

def max_dd(r):
    cum = np.exp(np.cumsum(r))
    rm  = np.maximum.accumulate(cum)
    return ((cum - rm) / rm).min()

def cvar95(r):
    var = np.percentile(r, 5)
    return r[r <= var].mean()



# ===== Cell 43: code =====

# all 7 strategies
all_strategies = {
    'Kelly (baseline)':       w_kelly,
    'Double-Kelly':           2.0 * w_kelly,
    'Half-Kelly':             0.5 * w_kelly,
    'Denoised-Kelly':         w_denoised_kelly,
    'Detoned-Kelly':          w_detoned_kelly,
    'Denoised+Detoned-Kelly': w_denoised_detoned_kelly,
    'HRP':                    w_hrp,
}



# ===== Cell 44: code =====


rows = []
oos_r = {}

for name, w in all_strategies.items():
    r = port_returns(w, returns_test)
    oos_r[name] = r
    rows.append({
        'Strategy':            name,
        'Cum. Return (%)':     round(cum_ret(r)*100, 4),
        'Ann. Sharpe':         round(ann_sharpe(r), 4),
        'Sortino':             round(sortino(r), 4),
        'Max Drawdown (%)':    round(max_dd(r)*100, 4),
        '95%-CVaR (daily, %)': round(cvar95(r)*100, 4),
    })

results_s6 = pd.DataFrame(rows).set_index('Strategy')
print("=" * 72)



# ===== Cell 45: code =====

print("  STEP 6 – OOS PERFORMANCE (Oct 1 – Dec 31, 2025)")
print("=" * 72)
display(results_s6)
print("=" * 72)
print("Note: Max Drawdown and CVaR are negative (losses).")



# ===== Cell 46: code =====





# ===== Cell 47: code =====

# Visualize it
colors_map = {
    'Kelly (baseline)':       'steelblue',
    'Double-Kelly':           'darkorange',
    'Half-Kelly':             'forestgreen',
    'Denoised-Kelly':         'crimson',
    'Detoned-Kelly':          'purple',
    'Denoised+Detoned-Kelly': 'brown',
    'HRP':                    'teal',
}

cum_df = pd.DataFrame(
    {name: np.exp(np.cumsum(r)) - 1 for name, r in oos_r.items()},
    index=returns_test.index
)



# ===== Cell 48: code =====

fig, ax = plt.subplots(figsize=(13, 6))
for col in cum_df.columns:
    ax.plot(cum_df.index, cum_df[col]*100, label=col,
            linewidth=2, color=colors_map[col])
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
ax.set_title('Step 6 – Cumulative Returns: All Strategies (Oct–Dec 2025)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Return (%)')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('step6_cumulative_returns.png', dpi=150)
plt.show()
print('Figure saved: step6_cumulative_returns.png')



# ===== Cell 49: code =====

# ── Metrics bar chart ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
bar_cols = [colors_map[s] for s in results_s6.index]
metric_map = [
    ('Cum. Return (%)',     'Cumulative Return (%)'),
    ('Ann. Sharpe',         'Annualised Sharpe Ratio'),
    ('Max Drawdown (%)',    'Maximum Drawdown (%)'),
]
for ax, (col, title) in zip(axes, metric_map):
    ax.bar(range(len(results_s6)), results_s6[col].values,
           color=bar_cols, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(results_s6)))
    ax.set_xticklabels(results_s6.index, rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(title, fontweight='bold', fontsize=10)
plt.suptitle('Step 6 – Performance & Risk Comparison (Oct–Dec 2025)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('step6_metrics_bar.png', dpi=150)
plt.show()
print('Figure saved: step6_metrics_bar.png')



# ===== Cell 50: code =====



