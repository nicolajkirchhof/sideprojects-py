#%%
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

import finance.utils as utils
import calendar

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters
mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%
symbol = 'QQQ'

df_barchart = utils.swing_trading_data.SwingTradingData(symbol, datasource='barchart')
# df_dolt = utils.swing_trading_data.SwingTradingData(symbol)

#%%

fig, axs = plt.subplots(3, 3, figsize=(24, 14))
# Average daily stats
df = df_barchart.df_day[['gappct', 'pct']].copy().reset_index(drop=True)
df['pct_week'] = df_barchart.df_week['pct'].copy().reset_index(drop=True)
df['pct_month'] = df_barchart.df_month['pct'].copy().reset_index(drop=True)
utils.plots.violinplot_columns_with_labels(df, ax=axs[0, 0])

# Dist from MAs
df = df_barchart.df_day[utils.definitions.EMA_DISTS]
utils.plots.violinplot_columns_with_labels(df, ax=axs[0, 1])

# ATRps
df = df_barchart.df_day[utils.definitions.ATRPs]
utils.plots.violinplot_columns_with_labels(df, ax=axs[1, 0])

# HVs
df = df_barchart.df_day[utils.definitions.HVs]
utils.plots.violinplot_columns_with_labels(df, ax=axs[1, 1])

# IVs
df = df_barchart.df_day[['iv', 'iv_pct', 'iv_rank']]
utils.plots.violinplot_columns_with_labels(df, ax=axs[2, 0])

# Vols
df_barchart.df_day['v10'] = df_barchart.df_day['v'] / 10
df = df_barchart.df_day[['v10', 'tot_oi', 'opt_vol']]
utils.plots.violinplot_columns_with_labels(df, ax=axs[2, 1])

df = df_barchart.df_day[['streak']].copy().reset_index(drop=True).rename(columns={'streak': 'streak_day'})
df['streak_week'] = df_barchart.df_week['streak'].copy().reset_index(drop=True)
df['streak_month'] = df_barchart.df_month['streak'].copy().reset_index(drop=True)
utils.plots.violinplot_columns_with_labels(df, ax=axs[0, 2])

# P/C interest/volume
df = df_barchart.df_day[['pc_oi', 'pc_vol']]
utils.plots.violinplot_columns_with_labels(df, ax=axs[1, 2])

# Slopes
df = df_barchart.df_day[utils.definitions.EMA_SLOPES]
utils.plots.violinplot_columns_with_labels(df, ax=axs[2, 2])

plt.show()

#%% Consecutive Up-Down days/weeks/month
utils.plots.plot_probability_tree(df_barchart.df_day['pct'], depth=6, title='SPY Daily Moves')
utils.plots.plot_probability_tree(df_barchart.df_week['pct'], depth=6, title='SPY Weekly Moves')
utils.plots.plot_probability_tree(df_barchart.df_month['pct'], depth=6, title='SPY Month Moves')

#%% Plot the percentage violins per month over the last 20 years
step = 4
start_year = df_barchart.df_day.year.min()
end_year = df_barchart.df_day.year.max()
total_range = range(start_year, end_year+1, step)
for year in total_range:
  year_range = range(year, year+step)
  fig, axs = plt.subplots(len(year_range), 1, figsize=(24, 14))

  for nax, year in enumerate(year_range):
    df = pd.DataFrame()
    for i in range(1, 13):
      df[f'{calendar.month_name[i]}'] = df_barchart.df_day[(df_barchart.df_day.month == i) & (df_barchart.df_day.year == year)]['pct'].reset_index(drop=True)
    utils.plots.violinplot_columns_with_labels(df, ax=axs[nax], title=f'{year}')

plt.show()

#%% Plot the percentage violins per day of the week over the last 20 years
step = 8
start_year = df_barchart.df_day.year.min()
end_year = df_barchart.df_day.year.max()
total_range = range(start_year, end_year+1, step)

for start_range_year in total_range:
  year_range = range(start_range_year, min(start_range_year + step, end_year + 1))
  num_rows = (len(year_range) + 1) // 2
  fig, axs = plt.subplots(num_rows, 2, figsize=(24, num_rows * 4))
  axs = axs.flatten()

  for nax, year in enumerate(year_range):
    df_day_plot = pd.DataFrame()
    # pandas .dt.dayofweek is 0=Mon, 6=Sun
    # Assuming df_day index is datetime, otherwise we use the 'dayofweek' column if available
    df_year = df_barchart.df_day[df_barchart.df_day.year == year]
    day_names = list(calendar.day_abbr) # Mon..Sun

    for i in range(5): # Trading days Mon-Fri
      # Use index.dayofweek if index is datetime, else adapt to your column
      df_day_plot[day_names[i]] = df_year[df_year.index.dayofweek == i]['pct'].reset_index(drop=True)

    utils.plots.violinplot_columns_with_labels(df_day_plot, ax=axs[nax], title=f'Daily Returns - {year}')

  for i in range(len(year_range), len(axs)):
    axs[i].set_visible(False)
  plt.tight_layout()

plt.show()

#%% Evaluate for all time profitable days

df_day_plot = pd.DataFrame()
day_names = list(calendar.day_abbr) # Mon..Sun

for i in range(5): # Trading days Mon-Fri
  # Use index.dayofweek if index is datetime, else adapt to your column
  df_day_plot[day_names[i]] = df_barchart.df_day[df_barchart.df_day.index.dayofweek == i]['pct'].reset_index(drop=True)

utils.plots.violinplot_columns_with_labels(df_day_plot, title=f'Daily Returns')

plt.show()

#%% Analyze the drawdowns from ATH with severity, duration, iv expansion in percentage
df_dd = df_barchart.df_day.copy()
if 'vwap3' not in df_dd.columns:
  df_dd['vwap3'] = (df_dd['h'] + df_dd['l'] + df_dd['c']) / 3

df_dd['ath'] = df_dd['c'].cummax()
df_dd['drawdown_pct'] = (df_dd['c'] - df_dd['ath']) / df_dd['ath'] * 100
df_dd['is_dd'] = df_dd['c'] < df_dd['ath']
# Group consecutive drawdown days
df_dd['dd_group'] = (df_dd['is_dd'] != df_dd['is_dd'].shift()).cumsum()

# Filter only for groups that are actually in drawdown
drawdown_groups = df_dd[df_dd['is_dd']].groupby('dd_group')

# 3. Visualization
fig = plt.figure(figsize=(24, 28))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 1.5])

ax_scatter = fig.add_subplot(gs[0])
ax_short   = fig.add_subplot(gs[1])
ax_med     = fig.add_subplot(gs[2])
ax_long    = fig.add_subplot(gs[3])

dd_summary = []
counts = {'short': 0, 'med': 0, 'long': 0}

for _, group in drawdown_groups:
    duration = len(group)
    if duration < 2: continue

    # Stats for scatter
    severity = group['drawdown_pct'].min()
    pre_dd_slice = df_dd.loc[df_dd.index < group.index[0], 'iv'].tail(1).values
    iv_bottom = group.loc[group['drawdown_pct'].idxmin(), 'iv']
    iv_exp = ((iv_bottom - pre_dd_slice[0]) / pre_dd_slice[0] * 100) if (len(pre_dd_slice) > 0 and pre_dd_slice[0] > 0) else 0

    dd_summary.append({'dur': duration, 'sev': abs(severity), 'iv': iv_exp})

    # Path normalization
    pre_dd = df_dd.loc[df_dd.index < group.index[0]].tail(1)
    base_price = pre_dd['vwap3'].values[0] if not pre_dd.empty else group['vwap3'].iloc[0]
    path = (group['vwap3'].to_numpy() / base_price - 1) * 100
    days = np.arange(len(path))

    if duration < 30:
        ax_short.plot(days, path, alpha=0.3, linewidth=1, color='tab:blue')
        counts['short'] += 1
    elif duration <= 65:
        ax_med.plot(days, path, alpha=0.5, linewidth=1.2, label=f"{group.index[0].year} ({duration}d)")
        counts['med'] += 1
    else:
        ax_long.plot(days, path, alpha=0.7, linewidth=1.5, label=f"{group.index[0].year} ({duration}d)")
        counts['long'] += 1

# 1. Scatter Plot
df_summ = pd.DataFrame(dd_summary)
sc = ax_scatter.scatter(df_summ['dur'], df_summ['sev'], c=df_summ['iv'], cmap='YlOrRd', s=100, alpha=0.6, edgecolors='black')
ax_scatter.set_title("Drawdown Severity vs Duration (Color: IV Expansion %)")
ax_scatter.set_xlabel("Duration (Days)")
ax_scatter.set_ylabel("Max Severity (%)")
fig.colorbar(sc, ax=ax_scatter, label="IV Expansion %")
ax_scatter.grid(True, alpha=0.2)

# 2. Path Overlays
for ax, key, title in [
    (ax_short, 'short', f"Short-Term (< 30 Days, n={counts['short']})"),
    (ax_med,   'med',   f"Medium-Term (30-90 Days, n={counts['med']})"),
    (ax_long,  'long',  f"Long-Term (> 90 Days, n={counts['long']})")
]:
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("VWAP3 % Change from ATH")
    ax.grid(True, alpha=0.2)

# Legends for labeled charts
if counts['med'] > 0:
    # Use a smaller font and multiple columns if there are many medium drawdowns
    ncol = 5 if counts['med'] > 15 else 3
    ax_med.legend(loc='lower left', fontsize=8, ncol=ncol, framealpha=0.6)

ax_long.set_xlabel("Days since ATH")
if counts['long'] > 0:
    ax_long.legend(loc='lower left', fontsize=9, ncol=4, framealpha=0.6)

plt.tight_layout()
plt.show()

#%% Analyze abnormal IV behavior and IV/HV relationships
df_vol = df_barchart.df_day.copy()
    
# Calculate daily changes and filter for available IV
df_vol = df_vol.dropna(subset=['iv']).copy()
df_vol['iv_change_pct'] = df_vol['iv'].pct_change() * 100

fig = plt.figure(figsize=(24, 22))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    
ax_iv_dyn = fig.add_subplot(gs[0, 0])
ax_premium = fig.add_subplot(gs[0, 1])
ax_hv_time = fig.add_subplot(gs[1, :]) # Span full width
ax_hv14 = fig.add_subplot(gs[2, 0])
ax_hv30 = fig.add_subplot(gs[2, 1])

# 1. IV Change % vs Price Change %
ax_iv_dyn.scatter(df_vol['pct'], df_vol['iv_change_pct'], 
                       c=np.abs(df_vol['pct']), cmap='Oranges', alpha=0.5)
ax_iv_dyn.axhline(0, color='black', lw=1); ax_iv_dyn.axvline(0, color='black', lw=1)
ax_iv_dyn.set_title("Volatility Dynamics: IV Change % vs Price Move %")
ax_iv_dyn.set_xlabel("Underlying Pct Change"); ax_iv_dyn.set_ylabel("IV Pct Change")
ax_iv_dyn.grid(True, alpha=0.2)

# 2. Volatility Premium Distribution (Top Right)
hv_cols = [c for c in df_vol.columns if c.startswith('hv')]
df_prem = pd.DataFrame({f'IV-{col}': df_vol['iv'] - df_vol[col] for col in hv_cols})
utils.plots.violinplot_columns_with_labels(df_prem, ax=ax_premium, title="IV - HV Premium Distributions")

# 3. Time Series (Filtered to IV availability)
ax_hv_time.plot(df_vol.index, df_vol['iv'], label='IV', color='black', lw=2)
for col in hv_cols:
    ax_hv_time.plot(df_vol.index, df_vol[col], label=col, alpha=0.4, lw=1)
ax_hv_time.set_title(f"IV vs Realized Volatilities (Range: {df_vol.index[0].date()} to {df_vol.index[-1].date()})")
ax_hv_time.legend(ncol=len(hv_cols)+1, loc='upper left')
ax_hv_time.grid(True, alpha=0.2)

# 4. IV vs HV14
max_v14 = max(df_vol['iv'].max(), df_vol['hv14'].max())
ax_hv14.scatter(df_vol['hv14'], df_vol['iv'], alpha=0.3, color='tab:blue')
ax_hv14.plot([0, max_v14], [0, max_v14], 'r--', alpha=0.6, label='IV=HV14')
ax_hv14.set_title("IV vs HV14 (Short-term Realized)")
ax_hv14.set_xlabel("HV14"); ax_hv14.set_ylabel("IV")
ax_hv14.legend(); ax_hv14.grid(True, alpha=0.2)

# 5. IV vs HV30 (From first evaluation)
# Ensure hv30 exists (assuming it follows your hv naming convention)
hv30_col = 'hv30' if 'hv30' in df_vol.columns else 'hv20' # Fallback if hv30 is missing
max_v30 = max(df_vol['iv'].max(), df_vol[hv30_col].max())
ax_hv30.scatter(df_vol[hv30_col], df_vol['iv'], alpha=0.3, color='tab:green')
ax_hv30.plot([0, max_v30], [0, max_v30], 'r--', alpha=0.6, label=f'IV={hv30_col}')
ax_hv30.set_title(f"IV vs {hv30_col.upper()} (Standard Realized)")
ax_hv30.set_xlabel(hv30_col.upper()); ax_hv30.set_ylabel("IV")
ax_hv30.legend(); ax_hv30.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

#%% Analyze abnormal IV behavior and IV/HV relationships
from scipy.stats import linregress
df_vol = df_barchart.df_day.copy()

# Filter for available IV and calculate dynamics
df_vol = df_vol[df_vol.iv > 0].dropna(subset=['iv', 'pct']).copy()
df_vol['iv_change_pct'] = df_vol['iv'].pct_change() * 100
df_vol = df_vol.dropna(subset=['iv_change_pct'])

fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1])

ax_iv_dyn  = fig.add_subplot(gs[0, 0:2])
ax_premium = fig.add_subplot(gs[0, 2:4])
ax_hv14    = fig.add_subplot(gs[1, 0])
ax_hv30    = fig.add_subplot(gs[1, 1])
ax_hv50    = fig.add_subplot(gs[1, 2])
ax_hv90    = fig.add_subplot(gs[1, 3])
ax_hv_time = fig.add_subplot(gs[2, :]) # Time series at the bottom

# 1. IV Change % vs Price Change % with Linear Regression
x_data = df_vol['pct']
y_data = df_vol['iv_change_pct']

# Calculate Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
line = slope * x_data + intercept

ax_iv_dyn.scatter(x_data, y_data, c=np.abs(x_data), cmap='YlOrRd', alpha=0.5, label='Data Points')
ax_iv_dyn.plot(x_data, line, color='darkblue', linewidth=2, label='Linear Fit')

# Plot the formula
formula_txt = f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}"
ax_iv_dyn.text(0.05, 0.95, formula_txt, transform=ax_iv_dyn.transAxes,
               fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax_iv_dyn.axhline(0, color='black', lw=1); ax_iv_dyn.axvline(0, color='black', lw=1)
ax_iv_dyn.set_title("Volatility Dynamics: IV Change % vs Price Move %")
ax_iv_dyn.set_xlabel("Underlying Pct Change"); ax_iv_dyn.set_ylabel("IV Pct Change")
ax_iv_dyn.legend()
ax_iv_dyn.grid(True, alpha=0.2)

# 2. Volatility Premium Distribution (Top Right)
hv_cols = [c for c in df_vol.columns if c.startswith('hv')]
df_prem = pd.DataFrame({f'IV-{col}': df_vol['iv'] - df_vol[col] for col in hv_cols})
utils.plots.violinplot_columns_with_labels(df_prem, ax=ax_premium, title="IV - HV Premium Distributions")

# 3. Scatters: IV vs HV14 and HV30 (Middle Row 1)
for ax, hv_type, color in [(ax_hv14, 'hv14', 'tab:blue'), (ax_hv30, 'hv30', 'tab:green'), (ax_hv50, 'hv50', 'tab:purple'), (ax_hv90, 'hv90', 'tab:purple')]:
    if hv_type not in df_vol.columns: continue
    limit = max(df_vol['iv'].max(), df_vol[hv_type].max())
    ax.scatter(df_vol[hv_type], df_vol['iv'], alpha=0.3, color=color)
    ax.plot([0, limit], [0, limit], 'r--', alpha=0.6, label=f'IV={hv_type.upper()}')
    ax.set_title(f"IV vs {hv_type.upper()}")
    ax.set_xlabel(hv_type.upper()); ax.set_ylabel("IV")
    ax.legend(); ax.grid(True, alpha=0.2)

# 5. Time Series (Bottom - Filtered)
ax_hv_time.plot(df_vol.index, df_vol['iv'], label='IV', color='black', linewidth=1)
for col in hv_cols:
    ax_hv_time.plot(df_vol.index, df_vol[col], label=col, alpha=0.5, linewidth=1.0)
ax_hv_time.set_title(f"Volatility Time Series (Filtered Range: {df_vol.index[0].date()} to {df_vol.index[-1].date()})")
ax_hv_time.legend(ncol=len(hv_cols)+1, loc='upper left', fontsize=9)
ax_hv_time.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

#%% Hurst Exponent, Autocorrelation, and Turn-of-the-Month Effect
df_stats = df_barchart.df_day.copy()

# 1. Calculate Rolling Hurst and Autocorrelation
# Hurst helper is already defined in indicators.py, using it here via apply
df_stats['hurst_100'] = df_stats['c'].rolling(window=100).apply(utils.indicators.hurst, raw=True)
df_stats['autocorr_1'] = df_stats['pct'].rolling(window=50).apply(lambda x: x.autocorr(lag=1))

# 2. End-of-Month / Beginning-of-Month (ToM) Labeling
# Identify the trading day of the month (1, 2, 3... and -1, -2, -3 from the end)
df_stats['day_of_month'] = df_stats.index.day

# Calculate business days from end of month
# We group by year/month and rank descending
df_stats['days_to_eom'] = df_stats.groupby([df_stats.index.year, df_stats.index.month])['day_of_month'].rank(ascending=False, method='first')

def label_tom_phase(row):
  if row['days_to_eom'] <= 3: return 'EOM (Last 3d)'
  if row['day_of_month'] <= 3: return 'BOM (First 3d)'
  return 'Mid-Month'

df_stats['month_phase'] = df_stats.apply(label_tom_phase, axis=1)

# 3. Visualization
fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 2)

ax_hurst = fig.add_subplot(gs[0, 0])
ax_ac    = fig.add_subplot(gs[0, 1])
ax_tom_v = fig.add_subplot(gs[1, :])
ax_tom_b = fig.add_subplot(gs[2, :])

# Plot Hurst over time
ax_hurst.plot(df_stats.index, df_stats['hurst_100'], color='tab:olive')
ax_hurst.axhline(0.5, color='black', linestyle='--', alpha=0.6, label='Random Walk')
ax_hurst.fill_between(df_stats.index, 0.5, df_stats['hurst_100'],
                      where=(df_stats['hurst_100'] > 0.5), color='green', alpha=0.1, label='Trending')
ax_hurst.fill_between(df_stats.index, 0.5, df_stats['hurst_100'],
                      where=(df_stats['hurst_100'] < 0.5), color='red', alpha=0.1, label='Mean-Reverting')
ax_hurst.set_title("Rolling 100-day Hurst Exponent")
ax_hurst.legend()

# Plot Autocorrelation
ax_ac.plot(df_stats.index, df_stats['autocorr_1'], color='tab:cyan')
ax_ac.axhline(0, color='black', lw=1)
ax_ac.set_title("Lag-1 Autocorrelation (50-day window)")
ax_ac.set_ylabel("Correlation Coeff")

# Turn-of-the-Month Distribution (Violin)
df_tom_pivot = pd.DataFrame()
for phase in ['BOM (First 3d)', 'Mid-Month', 'EOM (Last 3d)']:
  df_tom_pivot[phase] = df_stats[df_stats['month_phase'] == phase]['pct'].reset_index(drop=True)

utils.plots.violinplot_columns_with_labels(df_tom_pivot, ax=ax_tom_v, title="PnL % Distribution: Turn-of-the-Month Effect")

# Cumulative Returns by Phase
df_stats['cum_ret'] = df_stats['pct'].cumsum() # Simplified benchmark
for phase in df_stats['month_phase'].unique():
  # Mask returns not in phase and cumulate
  phase_rets = np.where(df_stats['month_phase'] == phase, df_stats['pct'], 0)
  ax_tom_b.plot(df_stats.index, np.cumsum(phase_rets), label=f"Only {phase}")

ax_tom_b.set_title("Cumulative Contribution to Returns by Monthly Phase")
ax_tom_b.legend()
ax_tom_b.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

#%% Hurst Exponent, Multi-Lag Autocorrelation, and 5-Day Monthly Windows
df_stats = df_barchart.df_day.copy()

# 1. Rolling Hurst
df_stats['hurst_100'] = df_stats['c'].rolling(window=100).apply(utils.indicators.hurst, raw=True)

# 2. Multi-Lag Autocorrelations (Legs)
# Lag 1: Daily momentum/mean-reversion
# Lag 5: Weekly cycle persistence
# Lag 21: Monthly cycle (Institutional window)
for lag in [1, 5, 21]:
  df_stats[f'ac_lag_{lag}'] = df_stats['pct'].rolling(window=60).apply(lambda x: x.autocorr(lag=lag))

# 3. Dissect Month into 5-Trading-Day Windows
# We calculate the ordinal trading day of the month (1, 2, 3...)
df_stats['trading_day_num'] = df_stats.groupby([df_stats.index.year, df_stats.index.month]).cumcount() + 1

def get_5day_window(d):
  if d <= 5:  return 'Days 1-5 (Open)'
  if d <= 10: return 'Days 6-10'
  if d <= 15: return 'Days 11-15 (Mid)'
  if d <= 20: return 'Days 16-20 (OpEx)'
  return 'Days 21+ (Close)'

df_stats['5d_window'] = df_stats['trading_day_num'].apply(get_5day_window)
window_order = ['Days 1-5 (Open)', 'Days 6-10', 'Days 11-15 (Mid)', 'Days 16-20 (OpEx)', 'Days 21+ (Close)']

# 4. Visualization
fig = plt.figure(figsize=(24, 20))
gs = fig.add_gridspec(4, 1)

ax_hurst = fig.add_subplot(gs[0, 0])
ax_ac    = fig.add_subplot(gs[1, 0])
ax_windows = fig.add_subplot(gs[2, 0])
ax_cum_win = fig.add_subplot(gs[3, 0])

# Hurst Plot
ax_hurst.plot(df_stats.index, df_stats['hurst_100'], color='tab:olive', lw=1.5)
ax_hurst.axhline(0.5, color='black', linestyle='--', alpha=0.6)
ax_hurst.set_title("Market Regime (Hurst 100d)")
ax_hurst.fill_between(df_stats.index, 0.5, df_stats['hurst_100'], where=(df_stats['hurst_100'] > 0.5), color='green', alpha=0.1)
ax_hurst.fill_between(df_stats.index, 0.5, df_stats['hurst_100'], where=(df_stats['hurst_100'] < 0.5), color='red', alpha=0.1)

# Multi-Lag Autocorrelation
for lag, col in [(1, 'tab:cyan'), (5, 'tab:purple'), (21, 'tab:gray')]:
  ax_ac.plot(df_stats.index, df_stats[f'ac_lag_{lag}'], label=f'Lag {lag}', color=col, alpha=0.7)
ax_ac.axhline(0, color='black', lw=1)
ax_ac.set_title("Autocorrelation Legs (Rolling 60d)")
ax_ac.legend(loc='upper left', ncol=3)

# 5-Day Window Distribution
df_win_pivot = pd.DataFrame()
for win in window_order:
  df_win_pivot[win] = df_stats[df_stats['5d_window'] == win]['pct'].reset_index(drop=True)

utils.plots.violinplot_columns_with_labels(df_win_pivot, ax=ax_windows, title="PnL % Distribution per 5-Trading-Day Window")

# Cumulative Performance by Window
for win in window_order:
  win_rets = np.where(df_stats['5d_window'] == win, df_stats['pct'], 0)
  ax_cum_win.plot(df_stats.index, np.cumsum(win_rets), label=win)

ax_cum_win.set_title("Cumulative Returns Contribution by Monthly Window")
ax_cum_win.legend(loc='upper left', ncol=3)
ax_cum_win.grid(True, alpha=0.2)

# Formatting X-Axis for all time-series plots in this section
for ax in [ax_hurst, ax_ac, ax_cum_win]:
  # Set major ticks to every month
  ax.xaxis.set_major_locator(mdates.YearLocator())
  # Format the label as Year-Month
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
  # Rotate labels to prevent overlap
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.show()

print("\nWindow Statistics (Mean Return %):")
print(df_stats.groupby('5d_window')['pct'].mean().reindex(window_order))
