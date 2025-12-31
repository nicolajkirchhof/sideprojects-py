#%%
import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import finance.utils as utils
import scipy.stats as stats

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters

# ... existing code ...
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
df_gaps = pd.read_pickle(f'finance/_data/all_gaps.pkl')


#%%
for i in range(-1, 21):
  df_gaps[f'rperf{i}'] = df_gaps[f'cpct{i}']- df_gaps[f'spy{i}']

# Replace inf and -inf with NaN
df_gaps.replace([np.inf, -np.inf], np.nan, inplace=True)
cpct_cols = df_gaps.filter(regex=r"^cpct-?\d+$").columns
c_cols = df_gaps.filter(regex=r"^c-?\d+$").columns
iv_cols = df_gaps.filter(regex=r"^iv-?\d+$").columns
hv_cols = df_gaps.filter(regex=r"^hv30-?\d+$").columns
spy_cols = df_gaps.filter(regex=r"^spy-?\d+$").columns
rvol_cols = df_gaps.filter(regex=r"^rvol-?\d+$").columns

# Pre-calculate categories before calling meta analysis
df_gaps= df_gaps[df_gaps.is_etf == 0].copy()
df_gaps['mcap_bin'] = df_gaps['market_cap'].map(utils.fundamentals.market_cap_classifier)
df_gaps['mcap_bin'] = pd.Categorical(df_gaps['mcap_bin'], categories=utils.fundamentals.MCAP_ORDER, ordered=True)

if 'c-1' in df_gaps.columns:
  df_gaps['price_bin'] = pd.qcut(df_gaps['c-1'], q=3, labels=['Low', 'Medium', 'High'])

# Prepare RVOL categories for meta analysis
if 'rvol0' in df_gaps.columns:
    rvol_bins = [0, 1, 3, 10, df_gaps['rvol0'].max()]
    rvol_labels = ['Low (<1x)', 'Normal (1-3x)', 'High (3-10x)', 'Extreme (>10x)']
    df_gaps['vol_category'] = pd.cut(df_gaps['rvol0'], bins=rvol_bins, labels=rvol_labels)

#%%
# df_gaps.to_pickle(f'finance/_data/all_gaps_w_stats.pkl')
df_gaps = pd.read_pickle(f'finance/_data/all_gaps_stats.pkl')
df_gaps = df_gaps.dropna(subset=list(c_cols))
df_stocks = df_gaps.copy()[df_gaps.is_etf == 0]
df_etfs = df_gaps.copy()[df_gaps.is_etf == 1]

#%%
# --- META ANALYSIS: Distribution and Time Evaluation ---
def plot_meta_analysis(df):
    # Create a larger grid to accommodate both directions
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(4, 2)
    
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    
    # Row 2: Gap UPS (> 0%)
    ax_up_mcap = fig.add_subplot(gs[1, 0])
    ax_up_price = fig.add_subplot(gs[1, 1])
    
    # Row 3: Gap DOWNS (< 0%)
    ax_dn_mcap = fig.add_subplot(gs[2, 0])
    ax_dn_price = fig.add_subplot(gs[2, 1])
    
    # Row 4: RVOL and Trends
    ax_rvol = fig.add_subplot(gs[3, 0])
    ax_trend = fig.add_subplot(gs[3, 1])

    # 1. Distribution by RVOL Category
    plot_range = (-30, 30)
    if 'vol_category' in df.columns:
        for cat in df['vol_category'].cat.categories:
            subset = df[df['vol_category'] == cat]
            ax_hist.hist(subset['gappct'].dropna(), bins=80, range=plot_range, alpha=0.5, label=f'RVOL: {cat}')
        ax_hist.set_title('Gap % Distribution by RVOL')
        ax_hist.legend()

    # 2. Time Count
    if 'date' in df.columns:
        df_time = df.copy()
        df_time['date'] = pd.to_datetime(df_time['date'])
        df_time.set_index('date').resample('ME').size().plot(ax=ax_time, marker='o', color='tab:blue')
        ax_time.set_title('Gap Frequency Over Time')

    # --- Directional Splitting Logic ---
    df_up = df[df['gappct'] > 0].copy()
    df_dn = df[df['gappct'] < 0].copy()

    # Helper to create wide DF for violin utility
    def get_wide_df(df_dir, group_col):
        categories = df[group_col].cat.categories
        dir_pivot = df_dir[df_dir['gappct'].between(-50, 50)].copy()
        return pd.DataFrame({cat: pd.Series(dir_pivot[dir_pivot[group_col] == cat]['gappct'].values) 
                             for cat in categories})

    # 3. UPS: MCAP & PRICE
    if 'mcap_bin' in df.columns:
        utils.plots.violinplot_columns_with_labels(get_wide_df(df_up, 'mcap_bin'), ax=ax_up_mcap, 
                                                  title='GAP UPS Magnitude by Market Cap')
    if 'price_bin' in df.columns:
        utils.plots.violinplot_columns_with_labels(get_wide_df(df_up, 'price_bin'), ax=ax_up_price, 
                                                  title='GAP UPS Magnitude by Price Level')

    # 4. DOWNS: MCAP & PRICE
    if 'mcap_bin' in df.columns:
        utils.plots.violinplot_columns_with_labels(get_wide_df(df_dn, 'mcap_bin'), ax=ax_dn_mcap, 
                                                  title='GAP DOWNS Magnitude by Market Cap')
    if 'price_bin' in df.columns:
        utils.plots.violinplot_columns_with_labels(get_wide_df(df_dn, 'price_bin'), ax=ax_dn_price, 
                                                  title='GAP DOWNS Magnitude by Price Level')

    # 5. RVOL Distribution
    ax_rvol.hist(df['rvol0'].dropna(), bins=100, range=(0, 15), color='teal', alpha=0.7)
    ax_rvol.set_title('Relative Volume Distribution (RVOL0)')

    # 6. Mean Trends
    if 'date' in df.columns:
        df_time.set_index('date').resample('ME')['gappct'].mean().plot(ax=ax_trend, color='tab:orange')
        ax_trend.axhline(0, color='black', lw=1, ls='--')
        ax_trend.set_title('Average Monthly Gap Sentiment')

    plt.tight_layout()
    plt.show()

# Run the meta analysis
plot_meta_analysis(df_gaps)

#%%
# --- STATISTICAL VALIDATION ---
def validate_grouping_factors(df):
  print("\n=== Statistical Factor Validation ===")

  # 1. Are Market Cap and Price Level independent?
  # (Chi-Square test on the bins)
  contingency_table = pd.crosstab(df['mcap_bin'], df['price_bin'])
  chi2, p_chi2, _, _ = stats.chi2_contingency(contingency_table)

  print(f"Independence Test (MCap vs Price): p-value = {p_chi2:.4f}")
  if p_chi2 < 0.05:
    print("-> FACTORS ARE LINKED: Low price stocks tend to cluster in specific market caps.")
  else:
    print("-> FACTORS ARE INDEPENDENT: Price level acts as a distinct variable.")

  # 2. Does Price Level significantly affect Gap Magnitude?
  # Using Kruskal-Wallis (Non-parametric ANOVA)
  price_groups = [group['gappct'].dropna() for name, group in df.groupby('price_bin', observed=True)]
  stat_p, p_val_p = stats.kruskal(*price_groups)

  # 3. Does Market Cap significantly affect Gap Magnitude?
  mcap_groups = [group['gappct'].dropna() for name, group in df.groupby('mcap_bin', observed=True)]
  stat_m, p_val_m = stats.kruskal(*mcap_groups)

  print(f"Kruskal-Wallis Effect of Price: p-value = {p_val_p:.4e}")
  print(f"Kruskal-Wallis Effect of MCap:  p-value = {p_val_m:.4e}")

  # 4. Which factor has more 'Explanatory Power'?
  # We can look at the Variance explained (eta-squared approximation)
  def eta_squared(groups):
    # Simple measure of effect size for Kruskal-Wallis
    h = stats.kruskal(*groups)[0]
    n = sum(len(g) for g in groups)
    return (h - len(groups) + 1) / (n - len(groups))

  eta_price = eta_squared(price_groups)
  eta_mcap = eta_squared(mcap_groups)

  print(f"Effect Size (Eta^2) - Price: {eta_price:.4f}")
  print(f"Effect Size (Eta^2) - MCap:  {eta_mcap:.4f}")

  if eta_price > eta_mcap:
    print("\nRESULT: Price Level is a STRONGER differentiator for gap size in this dataset.")
  else:
    print("\nRESULT: Market Cap is a STRONGER differentiator for gap size in this dataset.")

# Run validation on the stock subset
validate_grouping_factors(df_stocks)

#%%
# --- SIGNIFICANCE & NMR LOGIC ---
def classify_significance(row):
  score = 0
  # Magnitude
  if abs(row['gappct']) > 15: score += 3
  elif abs(row['gappct']) > 7: score += 2
  elif abs(row['gappct']) > 3: score += 1
  # Conviction
  if row['rvol0'] > 5: score += 3
  elif row['rvol0'] > 2: score += 2
  # Relative Strength vs SPY on Gap Day
  if abs(row['rperf0']) > 2: score += 1

  if score >= 5: return 'High'
  if score >= 3: return 'Medium'
  return 'Low'

df_gaps['significance'] = df_gaps.apply(classify_significance, axis=1)
#%%
def check_non_mean_reversion(row, days=20):
    """
    Checks if the price never returns to the breakout point (0%) 
    during the specified number of days post-gap.
    """
    t_cols = [f'cpct{i}' for i in range(0, days + 1) if f'cpct{i}' in row]
    if not t_cols: return np.nan

    vals = row[t_cols].values.astype(float)
    valid_vals = vals[np.isfinite(vals)]
    
    if valid_vals.size == 0: 
        return np.nan
        
    if row['gappct'] > 0:
        # For Gap Ups: The minimum price must stay above 0%
        return np.nanmin(valid_vals) > 0
    else:
        # For Gap Downs: The maximum price must stay below 0%
        return np.nanmax(valid_vals) < 0

# Calculate NMR for different timeframes
for d in [3, 7, 14, 20]:
    df_gaps[f'is_nmr_{d}d'] = df_gaps.copy().apply(check_non_mean_reversion, args=(d,), axis=1)

# Maintain the default column for backward compatibility with existing plots
df_gaps['is_nmr'] = df_gaps['is_nmr_20d']

#%%
# --- TIME SERIES ANALYSIS: Significance, NMR & SPY ---
def plot_nmr_evolution(df):
  df_t = df.copy()
  df_t['date'] = pd.to_datetime(df_t['date'])
  df_t = df_t.set_index('date')

  fig, ax1 = plt.subplots(figsize=(22, 10))
      
  # Group by significance to see success rate over time per class
  colors = {'High': 'tab:green', 'Medium': 'tab:orange', 'Low': 'tab:red'}
      
  for sig in ['High', 'Medium', 'Low']:
      sig_df = df_t[df_t['significance'] == sig]
      if sig_df.empty: continue
          
      monthly_sig = sig_df.resample('ME').agg({'is_nmr': 'mean'})
      ax1.plot(monthly_sig.index, monthly_sig['is_nmr'] * 100, 
               color=colors[sig], lw=2, label=f'NMR Rate - {sig} Significance', marker='o', alpha=0.8)

  ax1.set_ylabel('Non-Mean Reversion Rate (%)', fontsize=14)
  ax1.set_ylim(0, 10)

  # Plot SPY Baseline (Secondary Axis)
  ax2 = ax1.twinx()
  monthly_spy = df_t.resample('ME').agg({'spy0': 'mean'})
  ax2.bar(monthly_spy.index, monthly_spy['spy0'], width=15, alpha=0.2, color='gray', label='Avg SPY % (Gap Day)')
  ax2.set_ylabel('Avg SPY Day 0 Performance (%)', color='gray', fontsize=14)

  ax1.set_title('NMR Success Rate Over Time by Significance Level', fontsize=16)
  ax1.grid(True, alpha=0.3)
      
  # Combine legends
  lines, labels = ax1.get_legend_handles_labels()
  bars, bar_labels = ax2.get_legend_handles_labels()
  ax1.legend(lines + bars, labels + bar_labels, loc='upper left')

  plt.tight_layout()
  plt.show()

plot_nmr_evolution(df_gaps)

#%%
# --- CROSS-SECTIONAL ANALYSIS: Significance, Price & Market Cap vs NMR ---
def plot_nmr_factors(df):
  fig, axes = plt.subplots(1, 4, figsize=(28, 8))

  # 1. NMR Rate by Significance and Price (20d)
  pivot_price = df.pivot_table(index='significance', columns='price_bin', values='is_nmr_20d', aggfunc='mean')
  pivot_price.plot(kind='bar', ax=axes[0], color=['#ff9999','#66b3ff','#99ff99'])
  axes[0].set_title('20d NMR: Significance & Price')
  axes[0].set_ylabel('NMR Probability')

  # 2. NMR Rate by Significance and Market Cap (20d)
  pivot_mcap = df.pivot_table(index='significance', columns='mcap_bin', values='is_nmr_20d', aggfunc='mean')
  pivot_mcap.plot(kind='bar', ax=axes[1])
  axes[1].set_title('20d NMR: Significance & Market Cap')

  # 3. Success Rate Decay over Time
  decay_data = {}
  for d in [3, 7, 14, 20]:
      decay_data[f'{d}d'] = df.groupby('significance', observed=True)[f'is_nmr_{d}d'].mean()
  
  pd.DataFrame(decay_data).T.plot(ax=axes[2], marker='o')
  axes[2].set_title('NMR Success Rate Decay by Timeframe')
  axes[2].set_xlabel('Days Post-Gap')
  axes[2].set_ylabel('NMR Probability')
  axes[2].grid(True, alpha=0.3)

  # 4. Influence of RVOL on 20d NMR by Significance
  if 'vol_category' in df.columns:
    for sig in ['High', 'Medium', 'Low']:
        sig_df = df[df['significance'] == sig]
        sig_df.groupby('vol_category', observed=True)['is_nmr_20d'].mean().plot(
            kind='line', ax=axes[3], marker='s', label=sig)
        
    axes[3].set_title('20d NMR: RVOL & Significance')
    axes[3].set_ylabel('NMR Probability')
    axes[3].legend(title='Significance')
    axes[3].grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()

plot_nmr_factors(df_gaps)


#%%
# --- NMR INFLUENCE ANALYSIS (REVERSED EVALUATION) ---
def plot_nmr_feature_analysis(df):
  fig, axes = plt.subplots(2, 3, figsize=(24, 14))

  # 1. Influence of Earnings on NMR
  if 'earnings' in df.columns:
    # Group by NMR status and see % of them that were earnings gaps
    df_nmr = df.groupby('is_nmr', observed=True)['earnings'].mean()
    df_nmr.plot(kind='bar', ax=axes[0, 0], color=['salmon', 'skyblue'])
    axes[0, 0].set_title('Frequency of Earnings Gaps by NMR Status')
    axes[0, 0].set_xticklabels(['Mean Reverting', 'Non-Mean Reverting'], rotation=0)
    axes[0, 0].set_ylabel('% of Gaps that are Earnings')

  # 2. RVOL Distribution: NMR vs MR
  # This shows if higher volume actively prevents mean reversion
  if 'rvol0' in df.columns:
    df.boxplot(column='rvol0', by='is_nmr', ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_title('RVOL Distribution: MR vs NMR')
    axes[0, 1].set_xticklabels(['Mean Reverting', 'Non-Mean Reverting'])
    axes[0, 1].set_xlabel('')

  # 3. Gap Size Distribution: NMR vs MR
  # Does the size of the gap itself dictate stickiness?
  df['abs_gap'] = df['gappct'].abs()
  df.boxplot(column='abs_gap', by='is_nmr', ax=axes[0, 2], showfliers=False)
  axes[0, 2].set_title('Absolute Gap Size: MR vs NMR')
  axes[0, 2].set_xticklabels(['Mean Reverting', 'Non-Mean Reverting'])

  # 4. Market Cap Influence
  if 'mcap_bin' in df.columns:
    # Probability of NMR per Market Cap class
    df.groupby('mcap_bin', observed=True)['is_nmr'].mean().plot(kind='bar', ax=axes[1, 0], color='teal')
    axes[1, 0].set_title('NMR Probability by Market Cap')
    axes[1, 0].set_ylabel('Success Rate')

  # 5. Price Level Influence
  if 'price_bin' in df.columns:
    df.groupby('price_bin', observed=True)['is_nmr'].mean().plot(kind='bar', ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('NMR Probability by Price Bin')

  # 6. Combined Factor: Earnings + RVOL on NMR
  if 'earnings' in df.columns:
    pivot = df.pivot_table(index='vol_category', columns='earnings', values='is_nmr', aggfunc='mean')
    pivot.plot(ax=axes[1, 2], marker='o')
    axes[1, 2].set_title('NMR Rate: RVOL x Earnings')
    axes[1, 2].legend(['Non-Earnings', 'Earnings'])

  plt.suptitle('Factor Analysis: What Drives Non-Mean Reversion?', fontsize=20)
  plt.tight_layout()
  plt.show()

# Run the reversed evaluation
plot_nmr_feature_analysis(df_gaps)

#%%
# --- MACHINE LEARNING: Predicting NMR Sustainability ---
def predict_nmr_success(df):
  print("\n=== ML Prediction of Trend Sustainability (NMR) ===")

  # 1. Feature Preparation
  # Using the requested factors: is_etf, market_cap, earnings, rvol, relative performance (rperf0)
  features = ['is_etf', 'market_cap', 'earnings', 'rvol0', 'rperf0', 'gappct']

  # Filter for rows that have all features and are finite
  ml_df = df.dropna(subset=features + ['is_nmr_7d', 'is_nmr_14d', 'is_nmr_20d']).copy()

  # Convert earnings to int if it's boolean
  ml_df['earnings'] = ml_df['earnings'].astype(int)

  X = ml_df[features]
  targets = ['is_nmr_7d', 'is_nmr_14d', 'is_nmr_20d']

  feature_importances = pd.DataFrame(index=features)

  for target in targets:
    y = ml_df[target].astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    # Using class_weight='balanced' because NMR=True is usually the minority class
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    # Evaluation
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTarget: {target}")
    print(f"Model Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    feature_importances[target] = rf.feature_importances_

  # Visualize Feature Importance
  fig, ax = plt.subplots(figsize=(12, 8))
  feature_importances.plot(kind='bar', ax=ax)
  ax.set_title('Feature Importance for Predicting NMR across Timeframes')
  ax.set_ylabel('Importance Score')
  plt.tight_layout()
  plt.show()

predict_nmr_success(df_gaps)
#%%
def plot_gap_lines_df(gap_stats_df, alpha=0.2, lw=1.0, show_mean=True, title=''):
  """
  gap_stats_df: dataframe returned by gap_statistics(), containing columns:
    symbol, date, earnings, gappct, t-1, t0, ..., t14

  Plots each row's [t-1..t14] as y over x = 0..len(y)-1 (same behavior as before),
  plus an optional mean line.
  """
  fig, ax0 = plt.subplots(nrows=2, figsize=(23, 12))

  # Ensure we pick the window columns in the correct order
  t_cols = gap_stats_df.filter(regex=r"^t-?\d+$").columns

  ys = []
  for _, row in gap_stats_df.iterrows():
    y = row[t_cols].to_numpy(dtype=float)
    if y.size == 0:
      continue
    x = np.arange(y.size)
    ax0.plot(x, y, color="tab:blue", alpha=alpha, linewidth=lw)
    ys.append(y)

  if show_mean and ys:
    max_len = max(len(y) for y in ys)
    mat = np.full((len(ys), max_len), np.nan, dtype=float)
    for i, y in enumerate(ys):
      mat[i, :len(y)] = y
    mean_y = np.nanmean(mat, axis=0)
    ax0.plot(np.arange(max_len), mean_y, color="black", linewidth=2.5, label=f"Mean (n={len(ys)})")
    ax0.legend()

  # --- NEW: clamp y-axis to 5th..95th quantile (finite values only) ---
  vals = gap_stats_df.loc[:, t_cols].to_numpy(dtype=float).ravel()
  vals = vals[np.isfinite(vals)]
  if vals.size:
    y_lo, y_hi = np.quantile(vals, [0.05, 0.95])
    ax0.set_ylim(y_lo, y_hi)

  ax0.set_title(title)
  ax0.set_xlabel("Index (0..len(y)-1)")
  ax0.set_ylabel("Normalized % change (baseline = t-1)")
  ax0.grid(True, alpha=0.25)

  plt.tight_layout()
  return fig, ax0

#%%

# Group by Market Cap and Price levels
# Let's filter out ETFs first as they behave differently
df_stocks = df_gaps[df_gaps.is_etf == 0].copy()

# Use the refined classifier from fundamentals.py
# (Assuming you've imported it or defined it)
df_stocks['mcap_bin'] = df_stocks['market_cap'].map(utils.fundamentals.market_cap_classifier)

# Ensure correct sorting order for charts
mcap_order = ['Nano', 'Micro', 'Small', 'Mid', 'Large', 'Mega']
df_stocks['mcap_bin'] = pd.Categorical(df_stocks['mcap_bin'], categories=mcap_order, ordered=True)

# Create bins for Price (at the time of gap)
# Assuming 'c-1' or similar represents the price before the gap
if 'c-1' in df_stocks.columns:
  df_stocks['price_bin'] = pd.qcut(df_stocks['c-1'], q=3, labels=['Low (<$10)', 'Medium', 'High'])

# Analyze by Market Cap
for bin_name, group_df in df_stocks.groupby('mcap_bin', observed=True):
  print(f"Analyzing Market Cap: {bin_name} (n={len(group_df)})")
  fig, ax = boxplot_t_columns_with_labels(
    group_df,
    cpct_cols
  )
  ax.set_title(f"Price Change Distribution: {bin_name} Market Cap")
  plt.show()

# Analyze by Price Level
if 'price_bin' in df_stocks.columns:
  for bin_name, group_df in df_stocks.groupby('price_bin', observed=True):
    print(f"Analyzing Price Level: {bin_name} (n={len(group_df)})")
    boxplot_t_columns_with_labels(group_df, cpct_cols)
    plt.show()


#%%
# Analyze by Relative Volume (Conviction)
# Using rvol0 (relative volume on the gap day)
if 'rvol0' in df_stocks.columns:
  # Define meaningful volume thresholds
  # < 1.0: Below average, 1-3: Elevated, > 3: High Conviction
  rvol_bins = [0, 1, 3, df_stocks['rvol0'].max()]
  rvol_labels = ['Low Vol (<1x)', 'Elevated (1-3x)', 'High Conviction (>3x)']

  df_stocks['vol_category'] = pd.cut(
    df_stocks['rvol0'],
    bins=rvol_bins,
    labels=rvol_labels
  )

  for cat_name, group_df in df_stocks.groupby('vol_category', observed=True):
    if len(group_df) < 5: continue # Skip categories with too little data

    print(f"Analyzing Volume Category: {cat_name} (n={len(group_df)})")
    fig, ax = boxplot_t_columns_with_labels(
      group_df,
      regex=r"^cpct-?\d+$"
    )
    ax.set_title(f"Post-Gap Performance: {cat_name}")
    # Optional: Add a text box with the mean rvol for this group
    mean_v = group_df['rvol0'].mean()
    ax.text(0.02, 0.95, f"Avg RVOL: {mean_v:.2f}x", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.show()
#%%
max_diff = 75
gap_diffs = [3, 4] #, 5, 6, 7, 9, 11, 13, 15, 20, 30, 50, 100]
# Uncommon behavior
df_gap_uc = df_gaps[(df_gaps[t_cols] > 100).any(axis=1)]
for i, g_min in enumerate(gap_diffs[:-1]):
  g_max = gap_diffs[i+1]
  df_gaps_plt = df_gaps[(df_gaps.is_etf == 0) & (df_gaps.gappct >= g_min) & (df_gaps.gappct < g_max) & (df_gaps.t0 > g_min)]
  # print(f'{i}: {len(df_gaps_plt)}')
  # plot_gap_stats_df(df_gaps_plt, title=f'Gap Up {g_min}% - {g_max+1}%')
  boxplot_t_columns_with_labels(df_gaps_plt)
  plt.show()
#%%
# Evaluate gaps by pct above BOP and pct below BOP, mean / std performance
# symbol, date, gap%, t-1, t0, t1, t2, t3, t4, ..., t14
# Ensure we pick the window columns in the correct order
t_cols = df_gaps.filter(regex=r"^t-?\d+$").columns

# Uncommon behavior
df_gap_uc = df_gaps[(df_gaps[t_cols] > 100).any(axis=1)]

#%%
gap = next(gaps.itertuples())
for gap in gaps.itertuples():
  idx = gap[0]
  #%%
  aXdays = []
  for i in [1, 3, 5, 7, 10, 14]:
    data = df_stk.iloc[idx + i, :]
    data.rename(columns={'date': f'date{i}', 'o': f'o{i}', 'h': f'h{i}', 'c':f'c{i}', 'l':f'l{i}', 'v': f'v{i}', 'gap'})
    aXdays.append(data)
