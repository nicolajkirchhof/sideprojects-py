# %%
import itertools

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import finance.utils as utils
import seaborn as sns
import calendar
from glob import glob

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler # Better for pnl_pct outliers
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.feature_selection import mutual_info_regression

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
symbol = 'SPY'
# symbol = 'IWM'
# symbol = 'QQQ'

df_barchart = utils.swing_trading_data.SwingTradingData(symbol, datasource='barchart')


# df_dolt = utils.swing_trading_data.SwingTradingData(symbol)

## %%
def prepare_optionstrat_data(df):
  df = df.rename(columns={'P/L': 'pnl', 'P/L %': 'pnl_pct', 'Reason For Close': 'reason_close', 'Date Opened': 'date',
                          'Margin Req.': 'margin', 'Date Closed': 'date_closed'})[['pnl', 'pnl_pct', 'reason_close', 'date', 'margin', 'date_closed']]
  df['date'] = pd.to_datetime(df.date)
  df['date_closed'] = pd.to_datetime(df.date_closed)
  df.set_index('date', inplace=True)
  df = df[df.reason_close != 'Backtest Completed']
  return df


## %%
c_name = glob(f'finance/_data/optionsomega/{symbol}-C*.csv')
# df_ostrat_c = pd.read_csv('finance/_data/optionsomega/SPY-C-Delta-Band-45DTE-25D-5D-70D-PM.csv')
df_ostrat_c = pd.read_csv(c_name[0])


np_name = glob(f'finance/_data/optionsomega/{symbol}-NP*.csv')
# df_ostrat_np = pd.read_csv('finance/_data/optionsomega/SPY-NP-Delta-Band-45DTE-25D-50D-5D-PM.csv')
df_ostrat_np = pd.read_csv(np_name[0])

df_ostrat_c = prepare_optionstrat_data(df_ostrat_c)
df_ostrat_np = prepare_optionstrat_data(df_ostrat_np)

df_eval_c = pd.merge(df_ostrat_c.copy(), df_barchart.df_day.copy(), left_index=True, right_index=True, how='inner')
df_eval_np = pd.merge(df_ostrat_np.copy(), df_barchart.df_day.copy(), left_index=True, right_index=True, how='inner')

#%%
df_ostrat_nc = pd.read_csv('finance/_data/optionsomega/SPY-NC-Delta-Band-45DTE-20D-45D-10D-PM.csv')
df_ostrat_nc = prepare_optionstrat_data(df_ostrat_nc)
df_eval_nc = pd.merge(df_ostrat_nc.copy(), df_barchart.df_day.copy(), left_index=True, right_index=True, how='inner')
df_ostrat_p = pd.read_csv('finance/_data/optionsomega/SPY-P-Delta-Band-45DTE-25D-15D-75D-PM.csv')
df_ostrat_p = prepare_optionstrat_data(df_ostrat_p)
df_eval_p = pd.merge(df_ostrat_p.copy(), df_barchart.df_day.copy(), left_index=True, right_index=True, how='inner')

#%%
strategy_list = [
  (df_eval_c, "C Strategy"),
  (df_eval_np, "NP Strategy"),
  (df_eval_nc, "NC Strategy"),
  (df_eval_p, "P Strategy")
]
# %%
def eval_correlations_rational(df, name):
  df_corr = df[['pnl_pct', 'iv', 'iv_pct', 'gappct', *utils.definitions.HVs, *utils.definitions.EMA_DISTS,
                *utils.definitions.EMA_SLOPES, *utils.definitions.ATRPs]].corr()
  mask = np.triu(np.ones_like(df_corr, dtype=bool))
  utils.plots.heatmap(df_corr, mask, name)
  plt.show()


# %%

eval_correlations_rational(df_eval_c, 'C')
# eval_correlations_rational(df_eval_nc, 'NC')
eval_correlations_rational(df_eval_np, 'NP')
# eval_correlations_rational(df_eval_p, 'P')


# %%
def eval_correlations_discrete(df, name):
  for col in ['pnl_pct', *utils.definitions.EMA_DISTS, *utils.definitions.EMA_SLOPES]:
    df[f'{col}_cat'] = (df[col] > 0).astype(int)

  cols = df.filter(regex='.*_cat').columns
  df_corr = df[cols].corr()
  mask = np.triu(np.ones_like(df_corr, dtype=bool))
  utils.plots.heatmap(df_corr, mask, name)
  plt.show()


# %%
eval_correlations_discrete(df_eval_c, 'C')
# eval_correlations_discrete(df_eval_nc, 'NC')
eval_correlations_discrete(df_eval_np, 'NP')
# eval_correlations_discrete(df_eval_p, 'P')


# %%
def eval_non_linear_dependencies(df, target='pnl_pct', name=''):
  """Implements Mutual Information and MIC (Suggestions 3 & 5)"""
  features = ['iv', 'iv_pct', 'gappct', *utils.definitions.HVs, *utils.definitions.EMA_DISTS,
              *utils.definitions.EMA_SLOPES, *utils.definitions.ATRPs]

  # Ensure no NaNs for these calculations
  temp_df = df[[target] + features].dropna()
  X = temp_df[features]
  y = temp_df[target]

  # 1. Mutual Information (Suggestion 3)
  mi_scores = mutual_info_regression(X, y)
  mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)
  return mi_series
#%%

eval_non_linear_dependencies(df_eval_c, target='pnl_pct', name='C')
eval_non_linear_dependencies(df_eval_np, target='pnl_pct', name='NP')

# %%
# # Plotting results
# fig, ax1 = plt.subplots(figsize=(24, 12))
#
# mi_series.plot(kind='bar', ax=ax1, color='teal')
# ax1.set_title(f"Mutual Information Scores ({name})")
# ax1.set_ylabel("Dependency Score")
# plt.tight_layout()
# plt.show()

def plot_correlations(df, ref, name):
  features = ['iv', 'iv_pct', 'gappct', 'pc_oi', 'pc_vol', *utils.definitions.HVs, *utils.definitions.EMA_DISTS,
              *utils.definitions.EMA_SLOPES, *utils.definitions.ATRPs]
  for feature in features:
    plt.figure(figsize=(24, 12))
    plt.scatter(df[feature], df[ref])
    plt.xlabel(feature)
    plt.ylabel(ref)
    plt.title(f'{name}: {feature} vs {ref}')
    plt.show()


# %%
def pnl_per_year(df_eval, col, title):
  df = pd.DataFrame()
  df[col] = df_eval.pnl_pct
  for year in range(df_eval.index.min().year, df_eval.index.max().year + 1):
    df[f'{col}_{year}'] = df_eval.loc[df_eval.index.year == year, col]

  utils.plots.violinplot_columns_with_labels(df, title=title)
  plt.show()


# %%
pnl_per_year(df_eval_c, 'pnl_pct', 'C: PnL %')
pnl_per_year(df_eval_c, 'pnl', 'C: PnL')

pnl_per_year(df_eval_np, 'pnl_pct', 'NP: PnL %')
pnl_per_year(df_eval_np, 'pnl', 'NP: PnL')


# %% Strategy Permutation Analysis (Monte Carlo)
def run_strategy_permutations(df_eval, name, freq_weeks=[1, 2, 3, 4], n_sims=1000):
  """
  Permutates trades from the PnL distribution for different time frequencies.
  """
  # Ensure we have a clean series of PnL
  pnl_pool = df_eval['pnl'].dropna().to_numpy()

  results = []

  for wk in freq_weeks:
    # Number of trades per year based on frequency
    trades_per_year = max(1, int(52 / wk))

    for _ in range(n_sims):
      # Randomly sample trades for one "synthetic" year
      sim_pnl = np.random.choice(pnl_pool, size=trades_per_year, replace=True)

      # Calculate metrics for this simulation
      total_pnl = np.sum(sim_pnl)

      # Equity curve to calculate drawdown
      equity_curve = np.cumsum(np.insert(sim_pnl, 0, 0))
      running_max = np.maximum.accumulate(equity_curve)
      drawdowns = equity_curve - running_max
      max_dd = np.min(drawdowns)

      results.append({
        'frequency': f'{wk}wk',
        'total_pnl': total_pnl,
        'max_dd': max_dd,
        'is_positive': total_pnl > 0
      })

  df_sim = pd.DataFrame(results)

  # Plotting the variance
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

  # 1. PnL Distribution variance per frequency
  sns.violinplot(x='frequency', y='total_pnl', data=df_sim, ax=ax1, palette='viridis')
  ax1.set_title(f'{name}: Annual PnL Variance by Frequency')
  ax1.axhline(0, color='black', linestyle='--')

  # 2. Max Drawdown distribution
  sns.boxplot(x='frequency', y='max_dd', data=df_sim, ax=ax2, palette='magma')
  ax2.set_title(f'{name}: Max Drawdown Distribution')

  plt.tight_layout()
  plt.show()

  # Print statistics
  stats = df_sim.groupby('frequency').agg({
    'total_pnl': ['mean', 'std'],
    'max_dd': ['mean', 'min'],
    'is_positive': 'mean'
  })
  print(f"\nSimulation Stats for {name}:")
  print(stats)

# %% Execute for C and NP
run_strategy_permutations(df_eval_c, "C Strategy")
run_strategy_permutations(df_eval_np, "NP Strategy")

#%% Strategy Frequency Comparison Summary (Normalized)
def compare_selling_frequencies(strategies_dict, freq_weeks=[1, 2, 4]):
    """
    Plots a summary comparing different selling frequencies normalized
    to 'Average PnL per Trade' for better comparability.
    """
    num_strategies = len(strategies_dict)
    fig, axes = plt.subplots(num_strategies, 1, figsize=(24, 8 * num_strategies))
    if num_strategies == 1: axes = [axes]

    for i, (name, df_eval) in enumerate(strategies_dict.items()):
        df_daily = df_eval[['pnl']].resample('D').sum().fillna(0)

        yearly_results = []
        for wk in freq_weeks:
            days_step = wk * 5
            for offset in range(days_step):
                sampled = df_daily.iloc[offset::days_step]
                # We only count active trading days (non-zero pnl)
                active_trades = sampled[sampled['pnl'] != 0]

                for year, group in active_trades.groupby(active_trades.index.year):
                    num_trades = len(group)
                    if num_trades == 0: continue

                    # Normalize: Total PnL / Number of Trades in that year
                    avg_pnl_per_trade = group['pnl'].sum() / num_trades

                    yearly_results.append({
                        'frequency': f'{wk}wk',
                        'year': year,
                        'norm_pnl': avg_pnl_per_trade
                    })

        df_all = pd.DataFrame(yearly_results)
        df_bounds = df_all.groupby(['year', 'frequency']).agg(
            best=('norm_pnl', 'max'),
            worst=('norm_pnl', 'min')
        ).reset_index()

        years = sorted(df_bounds['year'].unique())
        x = np.arange(len(years))
        width = 0.8 / len(freq_weeks)

        for j, wk in enumerate(freq_weeks):
            label = f'{wk}wk'
            data = df_bounds[df_bounds['frequency'] == label].set_index('year').reindex(years)
            offset = (j - len(freq_weeks)/2 + 0.5) * width

            # Plot Normalized Best/Worst
            axes[i].bar(x + offset, data['best'], width, label=f'{label} Avg PnL (Best)', color=plt.cm.tab10(j), alpha=0.8)
            axes[i].bar(x + offset, data['worst'], width, label=f'{label} Avg PnL (Worst)', color=plt.cm.tab10(j), alpha=0.3, edgecolor='black', hatch='//')

        axes[i].set_title(f'Strategy: {name} - NORMALIZED Efficiency (Avg PnL per Trade)', fontsize=16)
        axes[i].set_ylabel("Avg PnL per Trade ($)")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(years)
        axes[i].axhline(0, color='black', linewidth=1)
        axes[i].legend(ncol=len(freq_weeks), loc='upper left')
        axes[i].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
# %% Execute for _c and _np
strategies = {
  "C (Calls)": df_eval_c,
  # "NC (Naked Calls)": df_eval_nc,
  "NP (Naked Puts)": df_eval_np,
  # "P (Puts)": df_eval_p
}

compare_selling_frequencies(strategies)
# %%
def analyze_best_trade_day(strategies_dict):
  """
  Analyzes and plots which day of the week (Mon-Fri) provides
  the best average and total PnL for each strategy.
  """
  num_strategies = len(strategies_dict)
  fig, axes = plt.subplots(num_strategies, 2, figsize=(24, 6 * num_strategies))
  if num_strategies == 1: axes = [axes]

  day_names = list(calendar.day_name)[:5]  # Monday to Friday

  for i, (name, df_eval) in enumerate(strategies_dict.items()):
    # Extract day of week (0=Mon, 4=Fri)
    df = df_eval.copy()
    df['day_of_week'] = df.index.dayofweek

    # Filter for standard trading days
    df = df[df['day_of_week'] <= 4]

    # Aggregate stats
    day_stats = df.groupby('day_of_week').agg({
      'pnl': ['mean', 'sum', 'count']
    })

    # 1. Plot Average PnL per Day
    axes[i, 0].bar(day_names, day_stats['pnl']['mean'], color='teal', alpha=0.7)
    axes[i, 0].set_title(f'{name}: Average PnL by Entry Day')
    axes[i, 0].set_ylabel("Mean PnL")
    axes[i, 0].axhline(0, color='black', linewidth=0.8)

    # 2. Plot Total Cumulative PnL per Day
    axes[i, 1].bar(day_names, day_stats['pnl']['sum'], color='darkblue', alpha=0.7)
    axes[i, 1].set_title(f'{name}: Total Cumulative PnL by Entry Day')
    axes[i, 1].set_ylabel("Total PnL")
    axes[i, 1].axhline(0, color='black', linewidth=0.8)

    # Add count labels on top of bars
    for j, count in enumerate(day_stats['pnl']['count']):
      axes[i, 1].text(j, day_stats['pnl']['sum'].iloc[j], f'n={count}',
                      ha='center', va='bottom' if day_stats['pnl']['sum'].iloc[j] > 0 else 'top', fontsize=9)

  plt.tight_layout()
  plt.show()


# %% Execute Day of Week Analysis

analyze_best_trade_day(strategies)

#%% Margin Requirement Analysis
def analyze_margin_utilization(df_eval, name):
  """
  Calculates the concurrent margin requirement by expanding
  the duration of each trade.
  """
  # Create a timeline of margin events
  margin_events = []
  for _, row in df_eval.iterrows():
    # Add margin at start
    margin_events.append({'time': row.name, 'change': row.margin})
    # Remove margin at end
    margin_events.append({'time': row.date_closed, 'change': -row.margin})

  df_events = pd.DataFrame(margin_events).sort_values('time')
  # Cumulative sum gives the total margin held at any timestamp
  df_events['total_margin'] = df_events['change'].cumsum()
  df_events.set_index('time', inplace=True)

  # Plotting
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))

  # 1. Margin Timeline
  df_events['total_margin'].plot(ax=ax1, color='purple', drawstyle='steps-post')
  ax1.set_title(f'{name}: Concurrent Margin Utilization Over Time')
  ax1.set_ylabel("Margin Required ($)")
  ax1.grid(True, alpha=0.3)

  # 2. Distribution of margin needs
  # Resample to daily max to get the peak margin required per day
  daily_margin = df_events['total_margin'].resample('D').max().fillna(0)
  utils.plots.violinplot_columns_with_labels(pd.DataFrame({'Peak Daily Margin': daily_margin}), ax=ax2)

  plt.tight_layout()
  plt.show()

  print(f"\nMargin Stats for {name}:")
  print(f"Max Absolute Margin: ${df_events['total_margin'].max():,.2f}")
  print(f"Average Margin Held: ${daily_margin.mean():,.2f}")

#%% Execute for _c and _np
analyze_margin_utilization(df_eval_c, "C Strategy")
analyze_margin_utilization(df_eval_np, "NP Strategy")

#%% IV Band Performance Analysis
def analyze_iv_performance_bands(df_eval, name):
  """
  Groups PnL by IV Percentage bands to see performance influence.
  Bands: <20, 20-50, 50-80, >80
  """
  df = df_eval.dropna(subset='iv_pct').copy()

  # Define the bands
  def get_iv_band(x):
    if x < .2: return '< .2'
    if x < .5: return '.2 - .5'
    if x < .8: return '.5 - .8'
    return '> .8'

  df['iv_band'] = df['iv_pct'].apply(get_iv_band)
  # Ensure correct ordering for plots
  band_order = ['< .2', '.2 - .5', '.5 - .8', '> .8']

  # 1. Plot PnL distribution per IV Band
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14))

  # Pivot for violin plot (one column per band)
  df_pivot = pd.DataFrame()
  for band in band_order:
    df_pivot[band] = df[df['iv_band'] == band]['pnl_pct'].reset_index(drop=True)

  utils.plots.violinplot_columns_with_labels(df_pivot, ax=ax1,
                                             title=f'{name}: PnL % Distribution by IV Band')

  # 2. Plot Total and Average PnL per Band over time
  # Group by Year and Band
  yearly_iv = df.groupby([df.index.year, 'iv_band'])['pnl_pct'].agg(['mean', 'sum', 'count']).unstack()

  # Clean up columns and handle missing data
  yearly_iv = yearly_iv.reindex(columns=band_order, level='iv_band')

  yearly_iv['sum'].plot(kind='bar', ax=ax2, alpha=0.8, cmap='viridis')
  ax2.set_title(f'{name}: Cumulative PnL % per Year grouped by IV Percentile Band')
  ax2.set_ylabel("Total PnL %")
  ax2.axhline(0, color='black', linewidth=1)
  ax2.legend(title="IV Pct Band")

  plt.tight_layout()
  plt.show()

  # Print detailed stats
  print(f"\nIV Band Statistics for {name}:")
  stats = df.groupby('iv_band')['pnl_pct'].agg(['count', 'mean', 'median', 'std', 'sum'])
  print(stats.reindex(band_order))

#%% Execute for _c and _np
analyze_iv_performance_bands(df_eval_c, "C Strategy")
analyze_iv_performance_bands(df_eval_np, "NP Strategy")


#%% Strategy Performance over Hurst and Autocorrelation Regimes
def analyze_regime_performance(df_eval, name):
  """
  Evaluates strategy PnL based on Hurst (Trending/Mean Reverting)
  and Autocorrelation (Momentum/Mean Reverting) regimes.
  """
  df = df_eval.copy()

  # 1. Define Regimes
  # Hurst: < 0.45 Mean Reverting, > 0.55 Trending, else Neutral
  df['hurst_regime'] = pd.cut(df['hurst'], bins=[0, 0.45, 0.55, 1.0],
                              labels=['Mean Reverting', 'Neutral', 'Trending'])

  # AC Lag 1: < 0 Mean Reverting, > 0 Momentum
  df['ac1_regime'] = np.where(df['ac_lag_1'] > 0, 'Momentum', 'Mean Reverting')

  # 2. Setup Plot
  fig, axes = plt.subplots(2, 2, figsize=(24, 16))

  # Top Left: PnL % vs Hurst Regime
  df_hurst_pivot = pd.DataFrame()
  for label in ['Mean Reverting', 'Neutral', 'Trending']:
    df_hurst_pivot[label] = df[df['hurst_regime'] == label]['pnl_pct'].reset_index(drop=True)
  utils.plots.violinplot_columns_with_labels(df_hurst_pivot, ax=axes[0, 0],
                                             title=f'{name}: PnL % by Hurst Regime')

  # Top Right: PnL % vs Autocorrelation (Lag 1)
  df_ac1_pivot = pd.DataFrame()
  for label in ['Mean Reverting', 'Momentum']:
    df_ac1_pivot[label] = df[df['ac1_regime'] == label]['pnl_pct'].reset_index(drop=True)
  utils.plots.violinplot_columns_with_labels(df_ac1_pivot, ax=axes[0, 1],
                                             title=f'{name}: PnL % by AC Lag 1 Regime')

  # Bottom Left: Heatmap of average PnL by Hurst + AC Lag 5 (Weekly Cycle)
  # Discretize Lag 5 for the matrix
  df['ac5_bin'] = pd.cut(df['ac_lag_5'], bins=[-1, -0.1, 0.1, 1], labels=['Neg AC5', 'Neutral', 'Pos AC5'])
  pivot_matrix = df.pivot_table(index='hurst_regime', columns='ac5_bin', values='pnl_pct', aggfunc='mean')
  sns.heatmap(pivot_matrix, annot=True, cmap='RdYlGn', center=0, ax=axes[1, 0], fmt=".2f")
  axes[1, 0].set_title(f"{name}: Mean PnL% (Hurst vs AC Lag 5)")

  # Bottom Right: Interaction with Lag 21 (Monthly Cycle)
  df['ac21_bin'] = np.where(df['ac_lag_21'] > 0, 'Pos AC21', 'Neg AC21')
  pivot_matrix_21 = df.pivot_table(index='hurst_regime', columns='ac21_bin', values='pnl_pct', aggfunc='mean')
  sns.heatmap(pivot_matrix_21, annot=True, cmap='RdYlGn', center=0, ax=axes[1, 1], fmt=".2f")
  axes[1, 1].set_title(f"{name}: Mean PnL% (Hurst vs AC Lag 21)")

  plt.tight_layout()
  plt.show()

  # Print Count summary to check sample sizes
  print(f"\nRegime Counts for {name}:")
  print(df.groupby(['hurst_regime', 'ac1_regime']).size().unstack())

#%% Execute Regime Analysis
analyze_regime_performance(df_eval_c, "C Strategy")
analyze_regime_performance(df_eval_np, "NP Strategy")

#%% Quantitative Regime Filtering Analysis (Hurst, AC5, AC21)
def evaluate_regime_improvement(df_eval, name):
  """
  Evaluates PnL improvement by filtering for specific technical regimes.
  Uses binary states for Hurst (<0.5), AC5 (>0), and AC21 (>0).
  """
  df = df_eval.copy().sort_index()

  # 1. Define Binary Regimes
  df['is_mr'] = df['hurst'] < 0.5  # Mean Reverting
  df['is_ac5_pos'] = df['ac_lag_5'] > 0
  df['is_ac21_pos'] = df['ac_lag_21'] > 0

  # Create a combined regime string for grouping
  df['regime_key'] = (
      np.where(df['is_mr'], "MR", "TR") + " | " +
      np.where(df['is_ac5_pos'], "AC5+", "AC5-") + " | " +
      np.where(df['is_ac21_pos'], "AC21+", "AC21-")
  )

  # 2. Calculate Group Statistics
  regime_stats = df.groupby('regime_key').agg(
    total_pnl=('pnl_pct', 'sum'),
    avg_pnl=('pnl_pct', 'mean'),
    occurrences=('pnl_pct', 'count')
  ).sort_values('avg_pnl', ascending=False)

  # Identify the 'Best' regime (Highest Average PnL with enough samples)
  best_regime = regime_stats[regime_stats['occurrences'] > 10].index[0]
  worst_regime = regime_stats.index[-1]
  negPnL_regimes = regime_stats[regime_stats.avg_pnl < 0].index

  # 3. Visualization
  fig = plt.figure(figsize=(24, 18))
  gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

  ax_equity = fig.add_subplot(gs[0, :])
  ax_counts = fig.add_subplot(gs[1, 0])
  ax_avg    = fig.add_subplot(gs[1, 1])

  # Plot A: Cumulative PnL Comparison
  df['cum_pnl_original'] = df['pnl_pct'].cumsum()
  # Filter: Only trades in the 'Best' regime
  df_filtered = df[df['regime_key'] != worst_regime].copy()
  df_filtered2 = df[~df['regime_key'].isin(negPnL_regimes)].copy()
  df['cum_pnl_filtered_worst'] = df_filtered['pnl_pct'].reindex(df.index).fillna(0).cumsum()
  df['cum_pnl_filtered_worst3'] = df_filtered2['pnl_pct'].reindex(df.index).fillna(0).cumsum()

  ax_equity.plot(df.index, df['cum_pnl_original'], label='Original Strategy (All Trades)', color='gray', alpha=0.6)
  ax_equity.plot(df.index, df['cum_pnl_filtered_worst'], label=f'Regime Filtered ({worst_regime})', color='tab:blue', lw=1.5)
  if not negPnL_regimes.empty:
    ax_equity.plot(df.index, df['cum_pnl_filtered_worst3'], label=f'Regime Filtered ({negPnL_regimes})', color='tab:purple', lw=1.5)
  ax_equity.set_title(f"{name}: Equity Curve Improvement via Regime Filtering")
  ax_equity.set_ylabel("Cumulative PnL %")
  ax_equity.legend()
  ax_equity.grid(True, alpha=0.2)

  # Plot B: Number of Occurrences
  regime_stats['occurrences'].plot(kind='barh', ax=ax_counts, color='tab:orange', alpha=0.7)
  ax_counts.set_title("Frequency of Regimes (Trade Counts)")
  ax_counts.set_xlabel("Number of Trades")

  # Plot C: Average PnL per Regime
  colors = ['green' if x > 0 else 'red' for x in regime_stats['avg_pnl']]
  regime_stats['avg_pnl'].plot(kind='barh', ax=ax_avg, color=colors, alpha=0.7)
  ax_avg.set_title("Average Edge (PnL %) per Regime")
  ax_avg.set_xlabel("Mean PnL %")
  ax_avg.axvline(0, color='black', lw=1)

  plt.tight_layout()
  plt.show()

  print(f"\nRegime Statistics for {name}:")
  print(regime_stats)
  print(f"\nRecommended Filter for {name}: {best_regime}")


#%% Statistical ML Influence Analysis
def analyze_indicator_influence_ml(df_eval, name):
  """
  Uses ML to rank indicators by their influence on PnL.
  Provides both Predictive Power (RF) and Directional Influence (Ridge).
  """
  # 1. Prepare Data
  # Define the indicator feature set
  features = ['iv', 'iv_pct', 'gappct', 'hurst', 'ac_lag_1', 'ac_lag_5', 'ac_lag_21','ema20_slope', 'ema50_slope', 'ema100_slope', 'ema200_slope',
              'ema10_dist', 'ema20_dist', 'ema50_dist', 'ema100_dist', 'ema200_dist', 'atrp14', 'atrp20', 'rvol']

  # Filter for rows where we have both PnL and all indicators
  df_ml = df_eval[[*features, 'pnl_pct']].dropna()
  if len(df_ml) < 50:
    print(f"Not enough data for ML analysis in {name}")
    return

  X = df_ml[features]
  y = df_ml['pnl_pct']

  # Scale features for the Linear model comparison
  # Use RobustScaler instead of StandardScaler to handle pnl_pct extremes
  scaler = RobustScaler()
  X_scaled = scaler.fit_transform(X)

  # 2. Train Models
  # Random Forest for non-linear predictive power
  rf = RandomForestRegressor(n_estimators=100, random_state=42)
  rf.fit(X, y)

  # Ridge (Linear) for directional influence
  ridge = Ridge(alpha=1.0)
  ridge.fit(X_scaled, y)

  # 3. Calculate Permutation Importance (More robust than default RF importance)
  perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

  # 4. Visualization
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

  # Plot A: Predictive Power (Permutation Importance)
  importance_df = pd.Series(perm_importance.importances_mean, index=features).sort_values()
  importance_df.plot(kind='barh', ax=ax1, color='tab:blue', alpha=0.7)
  ax1.set_title(f"{name}: Predictive Influence (Non-Linear Power)")
  ax1.set_xlabel("Importance Score (Mean Decrease in Accuracy)")

  # Plot B: Directional Influence (Ridge Coefficients)
  # Since data was scaled, coefficient magnitude represents strength
  coeff_df = pd.Series(ridge.coef_, index=features).sort_values()
  colors = ['green' if x > 0 else 'red' for x in coeff_df]
  coeff_df.plot(kind='barh', ax=ax2, color=colors, alpha=0.7)
  ax2.set_title(f"{name}: Directional Influence (Standardized Coeffs)")
  ax2.set_xlabel("Impact on PnL (Positive = Help, Negative = Hurt)")
  ax2.axvline(0, color='black', lw=1)

  plt.tight_layout()
  plt.show()

  # Include Mutual Information in the candidate selection 
  # to catch non-linear regime interactions
  mi_scores = mutual_info_regression(X, y, random_state=42)
  top_mi = pd.Series(mi_scores, index=features).nlargest(4).index.tolist()

  top_predictive = pd.Series(perm_importance.importances_mean, index=features).nlargest(6).index.tolist()
  top_directional = pd.Series(ridge.coef_, index=features).abs().nlargest(6).index.tolist()
      
  # Union now includes MI to ensure "Hidden Regime" candidates are captured
  candidate_union = list(set(top_predictive) | set(top_directional) | set(top_mi))

  print(f"\nRecommended Candidates for Permutation Analysis ({name}):")
  print(candidate_union)

  return candidate_union

#%% Execute ML Analysis
all_strategy_candidates = []

strategy_list = [
  (df_eval_c, "C Strategy"),
  (df_eval_np, "NP Strategy"),
  (df_eval_nc, "NC Strategy"),
  (df_eval_p, "P Strategy")
]

for df_eval, name in strategy_list:
  # Get candidates for this specific strategy
  strategy_candidates = analyze_indicator_influence_ml(df_eval, name)
  if strategy_candidates:
    all_strategy_candidates.extend(strategy_candidates)

##%% Consolidate and Print Final Candidate List
# Using a set to get unique features across all strategies
final_candidates = sorted(list(set(all_strategy_candidates)))

print("\n" + "="*50)
print("FINAL CONSOLIDATED CANDIDATES FOR SIMPLIFIED LOGIC")
print("="*50)
print(f"Total Unique Indicators: {len(final_candidates)}")
for i, feat in enumerate(final_candidates, 1):
  print(f"{i}. {feat}")
print("="*50)


#%% Logic Reduction for Simplified Trading Rules
def derive_simplified_trading_rules(regime_stats):
  """
  Simplifies boolean dependencies for both positive and negative regimes.
  Displays aggregated trade counts and average PnL for the simplified rules.
  """
  slots = [['MR', 'TR'], ['AC5+', 'AC5-'], ['AC21+', 'AC21-'], ['E100+', 'E100-'], ['V+', 'V-'], ['200+', '200-']]

  def run_reduction(target_df):
    # Convert DF rows to list of dicts for easier merging of stats
    # 'pattern' is the list of indicators, e.g., ['MR', 'AC5+', ...]
    current_rules = []
    for key, row in target_df.iterrows():
      current_rules.append({
        'pattern': key.split(' | '),
        'n': row['occurrences'],
        'sum_pnl': row['avg_pnl'] * row['occurrences']
      })

    def simplify_step(rules):
      new_rules = []
      used = set()
      for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
          r1, r2 = rules[i], rules[j]
          diff_idx = [idx for idx in range(len(r1['pattern'])) if r1['pattern'][idx] != r2['pattern'][idx]]

          if len(diff_idx) == 1:
            idx = diff_idx[0]
            val1, val2 = r1['pattern'][idx], r2['pattern'][idx]

            # Merge if they are opposite states in the same slot and not already wildcards
            if val1 != '*' and val2 != '*' and val1 in slots[idx] and val2 in slots[idx]:
              merged_pattern = list(r1['pattern'])
              merged_pattern[idx] = '*'

              merged_rule = {
                'pattern': merged_pattern,
                'n': r1['n'] + r2['n'],
                'sum_pnl': r1['sum_pnl'] + r2['sum_pnl']
              }
              # Avoid duplicate patterns in the next iteration
              if merged_rule['pattern'] not in [nr['pattern'] for nr in new_rules]:
                new_rules.append(merged_rule)
              used.add(i); used.add(j)

      unpaired = [rules[k] for k in range(len(rules)) if k not in used]
      return new_rules, unpaired

    all_final = []
    while True:
      combined, unique = simplify_step(current_rules)
      all_final.extend(unique)
      if not combined: break
      current_rules = combined

    # Clean up: Remove specific rules that are covered by more general (more wildcards) rules
    all_final.sort(key=lambda x: x['pattern'].count('*'), reverse=True)
    final_filtered = []
    for rule in all_final:
      is_subset = False
      for other in final_filtered:
        match = True
        for s in range(len(rule['pattern'])):
          if other['pattern'][s] != '*' and other['pattern'][s] != rule['pattern'][s]:
            match = False; break
        if match: is_subset = True; break
      if not is_subset: final_filtered.append(rule)
    return final_filtered

  # Process Positive and Negative groups
  pos_results = run_reduction(regime_stats[regime_stats['avg_pnl'] > 0])
  neg_results = run_reduction(regime_stats[regime_stats['avg_pnl'] <= 0])

  def print_report(title, results):
    print(f"\n--- {title} ---")
    # Sort results by absolute avg pnl
    for r in sorted(results, key=lambda x: abs(x['sum_pnl']/x['n']), reverse=True):
      active_terms = [t for t in r['pattern'] if t != '*']
      avg = r['sum_pnl'] / r['n']
      print(f"  IF [ {' | '.join(active_terms):<40} ] -> Avg: {avg:>6.2f}% | n: {int(r['n']):>3}")

  print_report("STRATEGIC EDGES (PROFITABLE REGIMES)", pos_results)
  print_report("RISK ZONES (NEGATIVE/TOXIC REGIMES)", neg_results)
  print("---------------------------------------")


#%% Quantitative Regime Filtering Analysis (Hurst, AC5, AC21)
def evaluate_regime_improvement(df_eval, name):
  """
   Evaluates PnL improvement by filtering for specific technical regimes.
   Adds EMA100 Slope, ATRP20 (Volatility), and EMA200 Distance.
   """
  df = df_eval.copy().sort_index()

  # 1. Define Binary Regimes
  df['is_mr'] = df['hurst'] < 0.5
  df['is_ac5_pos'] = df['ac_lag_5'] > 0
  df['is_ac21_pos'] = df['ac_lag_21'] > 0

  # New Indicators
  df['is_ema100_up'] = df['ema100_slope'] > 0
  # df['is_high_vol'] = df['atrp20'] > df['atrp20'].median() # Split by median volatility
  df['is_above_200'] = df['ema200_dist'] > 0

  # Create a combined regime string (Shortened for readability)
  df['regime_key'] = (
      np.where(df['is_mr'], "MR", "TR") + " | " +
      np.where(df['is_ac5_pos'], "AC5+", "AC5-") + " | " +
      np.where(df['is_ac21_pos'], "AC21+", "AC21-") + " | " +
      np.where(df['is_ema100_up'], "E100+", "E100-") + " | " +
      # np.where(df['is_high_vol'], "V+", "V-") + " | " +
      np.where(df['is_above_200'], "200+", "200-")
  )

  # 2. Calculate Group Statistics
  regime_stats = df.groupby('regime_key').agg(
    total_pnl=('pnl_pct', 'sum'),
    avg_pnl=('pnl_pct', 'mean'),
    occurrences=('pnl_pct', 'count')
  ).sort_values('avg_pnl', ascending=False)

  # Filter out regimes with very low sample sizes to avoid overfitting noise
  significant_regimes = regime_stats[regime_stats['occurrences'] >= 5]

  if significant_regimes.empty:
    print(f"No significant regimes found for {name} with current filters.")
    return

  best_regime = significant_regimes.index[0]
  worst_regime = significant_regimes.index[-1]
  negPnL_regimes = regime_stats[(regime_stats.avg_pnl < 0) | (regime_stats['occurrences'] < 5)].index

  # 3. Visualization
  fig = plt.figure(figsize=(24, 18))
  gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

  ax_equity = fig.add_subplot(gs[0, :])
  ax_counts = fig.add_subplot(gs[1, 0])
  ax_avg    = fig.add_subplot(gs[1, 1])

  # Plot A: Cumulative PnL Comparison
  df['cum_pnl_original'] = df['pnl_pct'].cumsum()
  # Filter: Only trades in the 'Best' regime
  df_filtered = df[df['regime_key'] != worst_regime].copy()
  df_filtered2 = df[~df['regime_key'].isin(negPnL_regimes)].copy()
  df['cum_pnl_filtered_worst'] = df_filtered['pnl_pct'].reindex(df.index).fillna(0).cumsum()
  df['negPnL_regimes'] = df_filtered2['pnl_pct'].reindex(df.index).fillna(0).cumsum()

  ax_equity.plot(df.index, df['cum_pnl_original'], label='Original Strategy (All Trades)', color='gray', alpha=0.6)
  if not negPnL_regimes.empty:
    ax_equity.plot(df.index, df['negPnL_regimes'], label=f'Regime Filtered ({negPnL_regimes})', color='tab:purple', lw=1.5)
  ax_equity.set_title(f"{name}: Equity Curve Improvement via Regime Filtering")
  ax_equity.set_ylabel("Cumulative PnL %")
  ax_equity.legend()
  ax_equity.grid(True, alpha=0.2)

  # Plot B: Number of Occurrences
  regime_stats['occurrences'].plot(kind='barh', ax=ax_counts, color='tab:orange', alpha=0.7)
  ax_counts.set_title("Frequency of Regimes (Trade Counts)")
  ax_counts.set_xlabel("Number of Trades")

  # Plot C: Average PnL per Regime
  colors = ['green' if x > 0 else 'red' for x in regime_stats['avg_pnl']]
  regime_stats['avg_pnl'].plot(kind='barh', ax=ax_avg, color=colors, alpha=0.7)
  ax_avg.set_title("Average Edge (PnL %) per Regime")
  ax_avg.set_xlabel("Mean PnL %")
  ax_avg.axvline(0, color='black', lw=1)

  plt.tight_layout()
  plt.show()

  print(f"\nRegime Statistics for {name}:")
  derive_simplified_trading_rules(regime_stats)

#%% Execute
evaluate_regime_improvement(df_eval_c, "C Strategy")
evaluate_regime_improvement(df_eval_np, "NP Strategy")
evaluate_regime_improvement(df_eval_nc, "NC Strategy")
evaluate_regime_improvement(df_eval_p, "P Strategy")

#%%
def analyze_feature_permutation_influence(df_eval, name, features, max_features=5):
    """
    Analyzes permutations using a Conservative Mean (Lower Bound).
    Applies logic reduction to show summarized trading conditions.
    """
    df = df_eval.copy().sort_index()
    results = []
    baseline_avg_pct = df['pnl_pct'].mean()
    total_trades_baseline = len(df)
    
    equity_curves = {'Baseline': df['pnl_pct'].cumsum()}

    # Thresholds to ensure robust influence factors
    MIN_REGIME_SAMPLES = 10
    # Minimum total trades for a logic set to be considered valid
    MIN_TOTAL_TRADES = 50

    for r in range(1, max_features + 1):
        print(f"Analyzing {name}: Combinations of size {r}...")
        group_results = []

        for combo in itertools.combinations(features, r):
            temp_df = df.copy()
            regime_cols = []
            slots = [] # For logic reduction

            for feat in combo:
                reg_name = f"reg_{feat}"
                if 'hurst' in feat:
                    states = ["MR", "TR"]
                    temp_df[reg_name] = np.where(temp_df[feat] < 0.45, "MR", "TR")
                elif any(x in feat for x in ['ac_lag', 'slope', 'dist']):
                    states = [f"{feat}+", f"{feat}-"]
                    temp_df[reg_name] = np.where(temp_df[feat] > 0, f"{feat}+", f"{feat}-")
                elif 'iv' in feat or 'atrp' in feat:
                    states = [f"{feat}H", f"{feat}L"]
                    temp_df[reg_name] = np.where(temp_df[feat] > temp_df[feat].median(), f"{feat}H", f"{feat}L")
                else:
                    states = [f"{feat}P", f"{feat}N"]
                    temp_df[reg_name] = np.where(temp_df[feat] > 0, f"{feat}P", f"{feat}N")
                regime_cols.append(reg_name)
                slots.append(states)

            temp_df['reg_key'] = temp_df[regime_cols].agg(' | '.join, axis=1)

            # Calculate stats including Standard Deviation for Standard Error calculation
            reg_stats = temp_df.groupby('reg_key')['pnl_pct'].agg(['mean', 'count', 'std']).fillna(0)

            # Penalize variance: Conservative Mean = Mean - StdError
            reg_stats['std_err'] = reg_stats['std'] / np.sqrt(reg_stats['count'])
            reg_stats['conservative_mean'] = reg_stats['mean'] - reg_stats['std_err']

            keep_regimes = reg_stats[(reg_stats['conservative_mean'] > 0) & (reg_stats['count'] >= MIN_REGIME_SAMPLES)].index
            df_filtered = temp_df[temp_df['reg_key'].isin(keep_regimes)]

            if len(df_filtered) >= MIN_TOTAL_TRADES:
                avg_conservative_eff = df_filtered['pnl_pct'].mean()

                # Perform Logic Reduction on the winning regimes
                current_rules = [{'pattern': k.split(' | '), 'n': 1, 'sum_pnl': 0} for k in keep_regimes]

                def simplify_step(rules, current_slots):
                    new_rules = []
                    used = set()
                    for i in range(len(rules)):
                        for j in range(i + 1, len(rules)):
                            r1, r2 = rules[i], rules[j]
                            diff_idx = [idx for idx in range(len(r1['pattern'])) if r1['pattern'][idx] != r2['pattern'][idx]]
                            if len(diff_idx) == 1:
                                idx = diff_idx[0]
                                if r1['pattern'][idx] != '*' and r2['pattern'][idx] != '*' and \
                                   r1['pattern'][idx] in current_slots[idx] and r2['pattern'][idx] in current_slots[idx]:
                                    merged_pattern = list(r1['pattern'])
                                    merged_pattern[idx] = '*'
                                    if merged_pattern not in [nr['pattern'] for nr in new_rules]:
                                        new_rules.append({'pattern': merged_pattern})
                                    used.add(i); used.add(j)
                    unpaired = [rules[k] for k in range(len(rules)) if k not in used]
                    return new_rules, unpaired

                reduced_rules = current_rules
                while True:
                    combined, unique = simplify_step(reduced_rules, slots)
                    if not combined: break
                    reduced_rules = combined + unique

                # Format using vertical dividers: MR | ac5+ | *
                logic_blocks = [" | ".join(r['pattern']) for r in reduced_rules]
                summarized_condition = " || ".join(logic_blocks)

                res_item = {
                    'feature_count': r,
                    'features': combo,
                    'condition': summarized_condition,
                    'avg_pnl_pct': avg_conservative_eff,
                    'trade_count': len(df_filtered),
                    'filter_rate': 1 - (len(df_filtered) / total_trades_baseline),
                    'score': avg_conservative_eff * np.sqrt(len(df_filtered)),
                    'equity_curve': df_filtered['pnl_pct'].reindex(df.index).fillna(0).cumsum()
                }
                results.append(res_item)
                group_results.append(res_item)

        if group_results:
            best_in_group = max(group_results, key=lambda x: x['score'])
            equity_curves[f'{r} Feat'] = best_in_group['equity_curve']

    df_res = pd.DataFrame(results).sort_values('score', ascending=False)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 18))
    sns.boxenplot(x='feature_count', y='avg_pnl_pct', data=df_res, hue='feature_count', palette='viridis', legend=False, ax=ax1)
    ax1.set_title(f"{name}: Conservative Efficiency Distribution")
    ax1.axhline(baseline_avg_pct, color='red', ls='--', label='Baseline Avg PnL %')

    for label, curve in equity_curves.items():
        lw = 3 if label == 'Baseline' else 1.5
        ax2.plot(curve.index, curve, label=label, linewidth=lw, alpha=0.8)
    ax2.set_title(f"{name}: Equity Comparison (Pruned to Robust Regimes)")
    ax2.set_ylabel("Cumulative PnL %")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    print(f"\n--- Simplified Trading Conditions for {name} ---")
    best_overall = df_res.sort_values(['feature_count', 'score'], ascending=[True, False]).groupby('feature_count').head(1)
    for _, row in best_overall.iterrows():
        print(f"SIZE {row['feature_count']}: {' + '.join(row['features'])}")
        print(f"  -> LOGIC: {row['condition']}")
        print(f"  -> Efficiency: {row['avg_pnl_pct']:.2f}% | Trades: {row['trade_count']} ({row['filter_rate']:.1%} filtered)")

    return df_res
# %% Execute permutation search using final_candidates
# final_candidates was defined in the previous block (ac_lag_1, hurst, ema200_dist, etc.)
# %% Execute Standard Evaluations for all strategies
for df_eval, name in strategy_list:
    eval_correlations_rational(df_eval, name)
    eval_correlations_discrete(df_eval, name)
    eval_non_linear_dependencies(df_eval, target='pnl_pct', name=name)

    pnl_per_year(df_eval, 'pnl_pct', f'{name}: PnL %')
    pnl_per_year(df_eval, 'pnl', f'{name}: PnL')

    run_strategy_permutations(df_eval, name)
    analyze_margin_utilization(df_eval, name)
    analyze_iv_performance_bands(df_eval, name)
    analyze_regime_performance(df_eval, name)
    evaluate_regime_improvement(df_eval, name)

# %% Execute Frequency Comparison Summary
strategies_dict = {name: df for df, name in strategy_list}
compare_selling_frequencies(strategies_dict)
analyze_best_trade_day(strategies_dict)

# %% Execute ML Influence Analysis and collect candidates
all_strategy_candidates = []

for df_eval, name in strategy_list:
    strategy_candidates = analyze_indicator_influence_ml(df_eval, name)
    if strategy_candidates:
        all_strategy_candidates.extend(strategy_candidates)

# Consolidate Final Candidate List
final_candidates = sorted(list(set(all_strategy_candidates)))

print("\n" + "="*50)
print("FINAL CONSOLIDATED CANDIDATES FOR SIMPLIFIED LOGIC")
print("="*50)
for i, feat in enumerate(final_candidates, 1):
    print(f"{i}. {feat}")
print("="*50)

# %% Execute permutation search using consolidated candidates
results = {}
for df_eval, name in strategy_list:
    results[name] = analyze_feature_permutation_influence(df_eval, name, final_candidates)
