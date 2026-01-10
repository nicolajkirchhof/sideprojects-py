import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import finance.utils as utils
import scipy.stats as stats

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %% [1] DATA LOADING & INITIAL CLEANING
df_atr2x = pd.read_pickle(f'finance/_data/all_atr2x.pkl')
df_atr2x = df_atr2x.reset_index(drop=True)
df_atr2x.replace([np.inf, -np.inf], np.nan, inplace=True)

# Ensure essential path columns exist
c_cols = df_atr2x.filter(regex=r"^c-?\d+$").columns
df_atr2x = df_atr2x.dropna(subset=list(c_cols))

# Robust Outlier Clipping for Numeric Data
numeric_cols = df_atr2x.select_dtypes(include=[np.number]).columns
for col_name in numeric_cols:
  upper_limit = df_atr2x[col_name].quantile(0.999)
  lower_limit = df_atr2x[col_name].quantile(0.001)
  df_atr2x[col_name] = df_atr2x[col_name].clip(lower_limit, upper_limit)

# %% [2] FEATURE ENGINEERING & DEFRAGMENTATION
# Calculate Relative Performance to SPY
new_cols = {}
for i in range(-1, 21):
  new_cols[f'rperf{i}'] = df_atr2x[f'cpct{i}'] - df_atr2x[f'spy{i}']
df_atr2x = pd.concat([df_atr2x, pd.DataFrame(new_cols, index=df_atr2x.index)], axis=1)
df_atr2x = df_atr2x.copy() # Defragment

# Define core feature columns to monitor
cols = ['c', 'cpct', 'v', 'atrp9', 'atrp14', 'atrp20', 'pct', 'rvol20', 'rvol50', 'iv', 'hv9', 'hv14', 'hv20', 'hv30', 'spy', 'rperf']

# Categorical Binning
df_atr2x['mcap_bin'] = df_atr2x['market_cap'].map(utils.fundamentals.market_cap_classifier)
df_atr2x['mcap_bin'] = pd.Categorical(df_atr2x['mcap_bin'], categories=utils.fundamentals.MCAP_ORDER, ordered=True)

if 'c-1' in df_atr2x.columns:
  df_atr2x['price_bin'] = pd.qcut(df_atr2x['c-1'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Standardize Volume Categories on rvol200
if 'rvol200' in df_atr2x.columns:
  rvol_bins = [0, 1, 3, 10, np.inf]
  rvol_labels = ['Low', 'Normal', 'High', 'Extreme']
  df_atr2x['vol_category'] = pd.cut(df_atr2x['rvol200'], bins=rvol_bins, labels=rvol_labels)

# Final Clean: Drop rows missing Day 0 features
valid_features = [f'{c}0' for c in cols if f'{c}0' in df_atr2x.columns]
df_atr2x = df_atr2x.dropna(subset=valid_features)

# Filter for Stocks only
df_stocks = df_atr2x[df_atr2x.is_etf == 0].copy()

# ATR-Intensity Features (Percentage move relative to daily ATR)
for window in [9, 14, 20]:
  atr_col = f'atrp{window}0'
  if atr_col in df_stocks.columns:
    df_stocks[f'actual_atr{window}x'] = (df_stocks['pct0'] / df_stocks[atr_col]).replace([np.inf, -np.inf], np.nan)

# %% [3] HIGH-INTENSITY & DIRECTIONAL NMR DEFINITION
# Filter for Moves > 200% ATR (Standard outlier criteria)
df_high = df_stocks[df_stocks['actual_atr14x'].abs() > 2.0].copy()

# IMPORTANT: Reset index here to prevent "Duplicate Label" errors during concatenation/joins
# This ensures each trade event is unique even if the same ticker moved on multiple days
df_high = df_high.reset_index(drop=True)

# Directional Non-Mean Reversion (NMR) logic
days_to_check = [f'cpct{i}' for i in range(1, 21)]
def calculate_nmr(row):
    path = row[days_to_check].values.astype(float)
    path = path[np.isfinite(path)]
    if len(path) == 0: return np.nan
    return np.min(path) > 0 if row['pct0'] > 0 else np.max(path) < 0

df_high['is_nmr'] = df_high.apply(calculate_nmr, axis=1)
df_high['direction'] = np.where(df_high['pct0'] > 0, 'Long', 'Short')

# Correct Market Support (SPY moves WITH stock)
df_high['spy_support'] = (np.sign(df_high['spy0']) == np.sign(df_high['pct0'])) & (df_high['spy0'] != 0)

# %% [4] REGIME & MULTI-FACTOR EVALUATION
# Bucket all continuous features for interaction analysis
bucket_cols = {}
for col in cols:
    feat = f'{col}0'
    if feat in df_high.columns:
        try:
            bucket_cols[f'{feat}_bin'] = pd.qcut(df_high[feat].abs(), q=3, labels=['L', 'M', 'H'], duplicates='drop')
        except ValueError: continue

if 'spy0' in df_high.columns:
    aligned_spy = df_high['spy0'] * np.sign(df_high['pct0'])
    bucket_cols['spy_alignment_bin'] = pd.qcut(aligned_spy, q=3, labels=['Against', 'Neutral', 'With'], duplicates='drop')
    move_spy = df_high['spy20'] * np.sign(df_high['pct0'])
    bucket_cols['spy_move_bin'] = pd.qcut(move_spy, q=3, labels=['Against', 'Neutral', 'With'], duplicates='drop')

df_buckets = pd.concat([df_high, pd.DataFrame(bucket_cols, index=df_high.index)], axis=1)

# Confluence Score (Earn + SPY Alignment + High Vol)
df_buckets['strength_score'] = (
    (df_buckets['earnings'].astype(int)) + 
    (df_buckets['spy_alignment_bin'] == 'With').astype(int) + 
    (df_buckets['rvol200_bin'] == 'H').astype(int)
)

# Profile Analysis
df_buckets['profile'] = (
    df_buckets['direction'] + "_" +
    df_buckets['rvol200_bin'].astype(str) + "_" +
    df_buckets['earnings'].map({True: 'EARN', False: 'TECH'})
)


# %% [5] INDEPENDENT FACTOR INFLUENCE ANALYSIS
# We iterate through all bucketed features to see their individual NMR probability
factor_cols = [c for c in df_buckets.columns if c.endswith('_bin')]
# Add specific non-bucketed binary/categorical factors
factor_cols += ['earnings', 'mcap_bin', 'price_bin', 'vol_category']

print("\n--- STANDALONE FACTOR INFLUENCE (NMR LIKELIHOOD) ---")

# Calculate how many rows we need for a grid of plots (3 plots per row)
n_factors = len(factor_cols)
n_cols = 3
n_rows = (n_factors + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 2.5 * n_rows))
axes = axes.flatten()

for i, factor in enumerate(factor_cols):
    if factor not in df_buckets.columns:
        continue

    stats = df_buckets.groupby(factor, observed=True)['is_nmr'].agg(['mean', 'count'])
    sns.barplot(x=stats.index, y=stats['mean'], ax=axes[i], palette='viridis')
    # # Calculate standalone stats for this factor
    # long_stats = df_buckets[df_buckets.direction == 'Long'].groupby(factor, observed=True)['is_nmr'].agg(['mean', 'count'])
    #
    # # Plot on the respective subplot
    # sns.barplot(x=long_stats.index, y=long_stats['mean'], ax=axes[i], palette='viridis')
    #
    # short_stats = df_buckets[df_buckets.direction == 'Short'].groupby(factor, observed=True)['is_nmr'].agg(['mean', 'count'])
    # sns.barplot(x=short_stats.index, y=short_stats['mean'], ax=axes[i], palette='viridis')

    # Add a baseline for comparison
    axes[i].axhline(df_high['is_nmr'].mean(), color='red', linestyle='--', alpha=0.6)
    
    # Formatting
    axes[i].set_title(f'Influence of: {factor}')
    axes[i].set_ylabel('NMR Probability')
    axes[i].set_ylim(0.5, stats['mean'].max() * 1.2 if not stats.empty else 1.0)
    axes[i].set_xlabel('')
    
    # # Add trade counts on top of bars
    for j, p in enumerate(axes[i].patches):
        idx = j % (len(stats.index)-1)
        axes[i].annotate(f'n={stats["count"].iloc[j]}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                         textcoords='offset points')

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Independent Factor Influence on Non-Mean Reversion Probability', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %% [5] INDEPENDENT FACTOR INFLUENCE ANALYSIS (SIDE-BY-SIDE LONG VS SHORT)
factor_cols = [c for c in df_buckets.columns if c.endswith('_bin') or c == 'spy_alignment_bin']
factor_cols += ['earnings', 'mcap_bin', 'price_bin', 'vol_category']
factor_cols = sorted(list(set(factor_cols)))

n_cols = 2
n_rows = (len(factor_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 2.5 * n_rows))
axes = axes.flatten()

for i, factor in enumerate(factor_cols):
  if factor not in df_buckets.columns: continue

  stats = df_buckets.groupby(factor, observed=True)['is_nmr'].agg(['mean', 'count'])
  # 1. Plot side-by-side bars for Long and Short
  sns.barplot(data=df_buckets, x=factor, y='is_nmr', hue='direction',
              ax=axes[i], palette={'Long': '#26a69a', 'Short': '#ef5350'},
              errorbar=None)

  # 2. Add sample size (n=) labels for each individual bar
  # We group by both factor and direction to get precise counts for each bar
  counts = df_buckets.groupby([factor, 'direction'], observed=True).size().unstack(fill_value=0)

  for j, category in enumerate(counts.index):
    # Annotate Long bar (left side of category tick)
    if 'Long' in counts.columns:
      n_long = counts.loc[category, 'Long']
      axes[i].annotate(f'n={n_long}', (j - 0.2, 0.05), ha='center',
                       fontsize=8, color='white', fontweight='bold', rotation=90)

    # Annotate Short bar (right side of category tick)
    if 'Short' in counts.columns:
      n_short = counts.loc[category, 'Short']
      axes[i].annotate(f'n={n_short}', (j + 0.2, 0.05), ha='center',
                       fontsize=8, color='white', fontweight='bold', rotation=90)
  # Add a baseline for comparison
  axes[i].axhline(df_high['is_nmr'].mean(), color='red', linestyle='--', alpha=0.6)

  # Formatting
  axes[i].set_title(f'Influence: {factor}', fontweight='bold')
  axes[i].set_ylabel('NMR Probability')
  axes[i].grid(axis='y', linestyle='--', alpha=0.3)
  axes[i].set_ylim(0.5, stats['mean'].max() * 1.2 if not stats.empty else 1.0)
  axes[i].legend(title='Direction', loc='upper right')
  axes[i].set_xlabel('')


# Cleanup
for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
plt.suptitle('Independent Factor Influence: Long vs Short Comparison', fontsize=20, y=1.01)
plt.tight_layout()
plt.show()

# %% [6] OUTPUT & VISUALIZATION
print(f"\n--- Directional Breakdown ({len(df_high)} trades) ---")
print(df_high.groupby('direction')['is_nmr'].agg(['mean', 'count']))

print("\n--- Top Success Profiles ---")
profile_stats = df_buckets.groupby('profile', observed=True)['is_nmr'].agg(['mean', 'count'])
print(profile_stats[profile_stats['count'] > 5].sort_values('mean', ascending=False).head(10))

# Confluence Point Plot
plt.figure(figsize=(10, 6))
sns.pointplot(data=df_buckets, x='strength_score', y='is_nmr', hue='direction', join=True, capsize=.1)
plt.title('NMR Probability by Confluence Score')
plt.ylabel('Continuation %')
plt.grid(axis='y', alpha=0.3)
plt.show()
