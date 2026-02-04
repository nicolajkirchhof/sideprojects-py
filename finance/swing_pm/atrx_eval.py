import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import finance.utils as utils

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %% [1] DATA LOADING & INITIAL CLEANING
df_atrx = pd.read_pickle(f'finance/_data/all_atr_x.pkl')
df_atrx = df_atrx.reset_index(drop=True)
df_atrx.replace([np.inf, -np.inf], np.nan, inplace=True)

# Ensure essential path columns exist
c_cols = df_atrx.filter(regex=r"^c-?\d+$").columns
df_atrx = df_atrx.dropna(subset=list(c_cols))

# Robust Outlier Clipping for Numeric Data
numeric_cols = df_atrx.select_dtypes(include=[np.number]).columns
for col_name in numeric_cols:
  upper_limit = df_atrx[col_name].quantile(0.999)
  lower_limit = df_atrx[col_name].quantile(0.001)
  df_atrx[col_name] = df_atrx[col_name].clip(lower_limit, upper_limit)

df_atrx = df_atrx.copy()
#%% Get the distribution of atrp changes
df_atrx['atrp_change'] = df_atrx['cpct0'] / df_atrx['atrp200']
atrp_change_clean = df_atrx['atrp_change'].replace([np.inf, -np.inf], np.nan).dropna()
lo, hi = atrp_change_clean.quantile([0.05, 0.95])
atrp_change_trimmed = atrp_change_clean[(atrp_change_clean >= lo) & (atrp_change_clean <= hi)]

atrp_change_trimmed.hist(bins=100)
plt.show()

#%%
# The dataset has the following columns 'symbol', 'earnings', 'date', 'gappct', 'c', 'is_etf', 'atrp20',
#   '1M', '1M_chg', '3M', '3M_chg', '6M', '6M_chg', 'market_cap', 'mcap_class', 'atrp_change'
# In addition, the following columns track changes before and after the event they are tracked
# as {name}XX and w_{name}XX for daily and weekly values before and after the event.
#   Daily columns are from -25 to 24 whereas 0 is the breakout day
#   Weekly columns are from -8 to 8 whereas 0 is the breakout day
# Tracked names
#   'c' => close, 'spy' => spy changes, v => volume, atrp9/14/20 => ATR percentage,
#   ac100_lag_1/5/20 => autocorrelation 100day lag 1/5/20, ac_comp => Composite Swing Signal (20-day, Avg Lags 1-3)
#   ac_mom => Standard Momentum (20-day, Lag-1), ac_mr => Short-Term Mean Reversion (10-day, Lag-1)
#   ac_inst => Institutional "Hidden" Momentum (60-day, Lag-5), pct => Percent Change, std_mv => 20 day standard deviation
#   rvol20/50 => Relative Volatility 20/50-day, iv => implied volatility,
#   hv9/14/20/50 => Historical Volatility 9/14/20/50-day, ema10/20_dist => EMA 10/20 distance
#   ema10/20_slope => EMA 10/20 slope, cpct => Percentage change in reference to breakout point

# What is the possibility of
# - staying above/below the 10/20 EMA if it was above/below it at c0
# - staying above/below the breakout point c-1
# - how much is the mean move away from the breakout point after 3, 5, 7, 10, 15, 20 days and 3, 4, 5, 6 weeks

# Influence factors for evaluations
# - Strength of breakout
# - Earnings
# - Market Cap
# - Performance 1M, 3M, 6M

# %% [2] META-ANALYSIS: Factors Influencing "Stay Above" Probabilities
# Factors: Market Cap Class, Breakout Strength (atrp_change), Earnings

# 2.1 Prepare Factors
meta_df = df_atrx.copy()

# Ensure ordered categorical if possible for cleaner plots
mcap_order = ['Micro-Cap', 'Small-Cap', 'Mid-Cap', 'Large-Cap']
meta_df['mcap_class_cat'] = pd.Categorical(meta_df['mcap_class'], categories=mcap_order, ordered=True)

# B) Breakout Strength (atrp_change) - Binned
if 'atrp_change' in meta_df.columns:
  s_atr = meta_df['atrp_change'].replace([np.inf, -np.inf], np.nan).abs() # Use absolute values
  # Clip extreme outliers for cleaner binning
  s_atr = s_atr.clip(s_atr.quantile(0.01), s_atr.quantile(0.99))
  meta_df['strength_bin'], bins = pd.qcut(s_atr, q=4, labels=['Weak', 'Moderate', 'Strong', 'Explosive'], retbins=True)
  
  print(f"\n--- Breakout Strength Class Boundaries (Absolute atrp_change) ---")
  print(f"Weak:      {bins[0]:.4f} to {bins[1]:.4f}")
  print(f"Moderate:  {bins[1]:.4f} to {bins[2]:.4f}")
  print(f"Strong:    {bins[2]:.4f} to {bins[3]:.4f}")
  print(f"Explosive: {bins[3]:.4f} to {bins[4]:.4f}")

# C) Earnings Event (Boolean)
if 'earnings' in meta_df.columns:
  meta_df['has_earnings'] = meta_df['earnings'].fillna(False).astype(bool)
  meta_df['earnings_label'] = meta_df['has_earnings'].map({True: 'Earnings', False: 'Non-Earnings'})

# 2.2 Define Success Metrics (Stay Above Logic)
FUTURE_DAYS = list(range(1, 21))

def calculate_stay_probs(sub_df):
  """Calculates probability of staying above key levels for a given subset."""
  if sub_df.empty: return pd.Series({'ema10': np.nan, 'ema20': np.nan, 'breakout': np.nan, 'count': 0})

  # EMA 10 Support (Only consider if STARTING above EMA10)
  ema10_start = sub_df['ema10_dist0'] > 0
  ema10_cols = [c for c in [f'ema10_dist{d}' for d in FUTURE_DAYS] if c in sub_df.columns]
  if ema10_cols and ema10_start.any():
    ema10_stay = (sub_df.loc[ema10_start, ema10_cols] > 0).all(axis=1).mean()
  else:
    ema10_stay = np.nan

  # EMA 20 Support (Only consider if STARTING above EMA20)
  ema20_start = sub_df['ema20_dist0'] > 0
  ema20_cols = [c for c in [f'ema20_dist{d}' for d in FUTURE_DAYS] if c in sub_df.columns]
  if ema20_cols and ema20_start.any():
    ema20_stay = (sub_df.loc[ema20_start, ema20_cols] > 0).all(axis=1).mean()
  else:
    ema20_stay = np.nan

  # Breakout Support (Stay above c0 price) - using cpct relative to breakout
  # cpct{d} is (price{d} - price_breakout) / price_breakout
  # So cpct > 0 means above breakout price
  bk_cols = [c for c in [f'cpct{d}' for d in FUTURE_DAYS] if c in sub_df.columns]
  # We filter for trades that actually broke out UP (pct0 > 0 or similar check usually implied)
  # Assuming the dataset contains breakouts, we check if they STAY above.
  if bk_cols:
    # Check if the move was initially positive or negative
    # We assume the direction is set by the initial move (cpct0)
    # If cpct0 is positive, we want future cpct to be > 0 (stay above)
    # If cpct0 is negative, we want future cpct to be < 0 (stay below / continuation)
    
    # Vectorized check:
    # If initial move (atrp_change) was positive, check if cpct > 0
    # If initial move was negative, check if cpct < 0
    # This is equivalent to checking if cpct has the same sign as atrp_change
    
    # Since sub_df might have mixed positive/negative breakouts, we need row-wise logic.
    # However, for simple "success" metric:
    # Success = (cpct{d} > 0) IF (atrp_change > 0)
    # Success = (cpct{d} < 0) IF (atrp_change < 0)
    # Combined: cpct{d} * atrp_change > 0
    
    # Let's align with the column 'atrp_change' which exists in meta_df and should exist in sub_df (grouping preserves it or we access via index)
    # Note: sub_df is a chunk of meta_df.
    
    direction = np.sign(sub_df['atrp_change'].values[:, None]) # Shape (N, 1)
    future_vals = sub_df[bk_cols].values # Shape (N, D)
    
    # Check if future values are in the same direction as the breakout
    # We use strict inequality > 0 to imply "continuing in that direction" or "staying on that side"
    is_continuing = (future_vals * direction) > 0
    
    bk_stay = is_continuing.all(axis=1).mean()
  else:
    bk_stay = np.nan

  return pd.Series({
    'Stay > EMA10': ema10_stay,
    'Stay > EMA20': ema20_stay,
    'Stay > Breakout': bk_stay,
    'count': len(sub_df)
  })

# 2.3 Aggregate & Plot
factors = ['mcap_class_cat', 'strength_bin', 'earnings_label']
results = {}

for factor in factors:
  if factor not in meta_df.columns: continue

  # Group by factor and calculate probabilities
  grouped = meta_df.groupby(factor, observed=True).apply(calculate_stay_probs, include_groups=False)

  if grouped.empty or grouped['count'].empty: continue

  # Filter out groups with very low sample size
  grouped = grouped[grouped['count'] > 10]

  if grouped.empty: continue

  # Plotting
  metrics = ['Stay > EMA10', 'Stay > EMA20', 'Stay > Breakout']

  # Melt for seaborn
  plot_data = grouped[metrics].reset_index().melt(id_vars=factor, var_name='Metric', value_name='Probability')

  plt.figure(figsize=(10, 5))
  ax = sns.barplot(data=plot_data, x=factor, y='Probability', hue='Metric', palette='viridis')

  plt.title(f'Probability of Holding Support Levels by {factor}')
  plt.ylabel('Probability (Day 1-20)')
  plt.ylim(0, 1.0)
  plt.grid(axis='y', linestyle='--', alpha=0.3)

  # Add count labels above groups (using the first bar of the group usually)
  # This is a bit tricky with hue, so we'll just print the counts to console for detailed view
  plt.tight_layout()
  plt.show()

  print(f"\n--- Detailed Stats for {factor} ---")
  print(grouped)
