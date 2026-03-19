# %%
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

import finance.utils as utils
import calendar

from finance.utils.move_character import (
    calculate_regime_filter_stats,
    calculate_move_magnitude_stats,
    calculate_intratrend_retracement,
    calculate_gap_intraday_decomposition,
    calculate_hv_regime_stats,
    calculate_impulse_forward_returns,
)
from finance.utils.plots import annotate_violin

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters
mpl.use('TkAgg')
mpl.use('QtAgg')
# %load_ext autoreload
# %autoreload 2

# %%
symbol = 'QQQ'

data = utils.SwingTradingData(symbol, datasource='offline')


# df_dolt = utils.swing_trading_data.SwingTradingData(symbol)

#%%
# Questions on stock
# - probability of two, three, four, five successive down/up days/weeks/month
#     Create a chart for each timerange
# - duration and percentage change following 5, 10, 20 MA daily and weekly.
#     Following is defined as the slope of the MA becoming positive / negative until it turns into the opposite direction.
#     Create violin plots for the daily and weekly evaluations in both positive and negative directions
#         trend following duration. Recalculate the weekly duration to days.
#         price changes
# - duration and percentage change of consecutive weekly
#     up / down moves defined as the close not being below / above the prior bar low / high
#     trend following the 5 MA when the 5 MA > 10 MA (weekly) or 5 MA > 10 MA > 20 MA (daily)
#     to the first close below the 5 MA.
#         The same with opposite MAs for the negative trend following
#     Create one violin plot for the durations and one for the percentage change combining both evaluations
# For all violin plots, annotate the 25, 50, 75 quantile
# Layout: 3 row grid
#     Row 1: Prob. of streak length, Weekly trend durations, Weekly trend price changes
#     Row 2: MA slope durations (Daily/Weekly, Pos/Neg)
#     Row 3: MA slope price changes (Daily/Weekly, Pos/Neg)
# Theme: Dark background
# Align the plots so that the space use is optimal

def calculate_streak_probabilities(df, label):
  """
  Calculate the probability of consecutive up/down streaks.
  """
  # streak is already in df from swing_indicators (via SwingTradingData)
  streaks = df['streak']

  # Find the end of each streak
  # Shift to find where the next value is different or it's the end
  streak_ends = streaks.iloc[np.where(streaks.values != np.roll(streaks.values, -1))[0]]
  if len(streaks) > 0:
    streak_ends = pd.concat([streak_ends, streaks.iloc[[-1]]])

  up_ends = streak_ends[streak_ends > 0]
  down_ends = streak_ends[streak_ends < 0].abs()

  total_up = len(up_ends)
  total_down = len(down_ends)

  max_streak = 6
  up_probs = {}
  down_probs = {}

  for i in range(2, max_streak):
    up_probs[i] = (up_ends >= i).sum() / total_up if total_up > 0 else 0
    down_probs[i] = (down_ends >= i).sum() / total_down if total_down > 0 else 0

  return up_probs, down_probs


def calculate_ma_slope_stats(df, ma_periods=[5, 10]):
  """
  Calculate duration and change following 5, 10 MA.
  Following is defined as the slope of the MA becoming positive / negative until it turns into the opposite direction.
  """
  stats = {}
  for p in ma_periods:
    slope = df[f'ma{p}_slope']
    positive_slope = slope > 0

    # Identify contiguous blocks
    # Drop first few rows where MA is NaN
    valid_mask = positive_slope.notna()
    ps = positive_slope[valid_mask]
    df_valid = df[valid_mask]

    blocks = (ps != ps.shift()).cumsum()

    # Group by blocks
    group = df_valid.groupby(blocks)

    durations = group.size()
    # For price change, we want the change from the prior close of the first bar to the last bar close
    # This includes the price movement of the bar that caused the slope to change.
    changes = group.apply(lambda x: (x['c'].iloc[-1] / df.iloc[df.index.get_loc(x.index[0]) - 1]['c'] - 1) * 100 if df.index.get_loc(x.index[0]) > 0 else np.nan)

    # To filter 'positive' blocks, we check the first value of 'positive_slope' for each block
    is_positive_block = ps.groupby(blocks).first()

    stats[p] = {
      'pos_dur': durations[is_positive_block].values,
      'pos_chg': changes[is_positive_block].values,
      'neg_dur': durations[~is_positive_block].values,
      'neg_chg': changes[~is_positive_block].values
    }
  return stats


def calculate_high_low_streak_stats(df):
  """
  Calculate duration and percentage change of consecutive weekly up / down moves
  defined as the close not being below / above the prior bar low / high.
  """
  # For 'Up' move: close >= prior low
  # For 'Down' move: close <= prior high
  # The requirement says "consecutive weekly up / down moves that are defined as the close not being below / above the prior bar low / high"
  # Let's interpret:
  # Up streak: close >= prior low
  # Down move: close <= prior high

  prev_h = df['h'].shift()
  prev_l = df['l'].shift()

  is_up = df['c'] >= prev_l
  is_down = df['c'] <= prev_h

  # Since a bar can't be both >= prev_h and <= prev_l (usually), but it could be neither.
  # If it's neither, the streak breaks.
  # We need to identify streaks of 'is_up' and streaks of 'is_down'.

  def get_streaks(condition):
    blocks = (condition != condition.shift()).cumsum()
    group = df.groupby(blocks)
    durations = group.size()
    # Use prior close for more accurate price change that includes the entry bar's move
    changes = group.apply(lambda x: (x['c'].iloc[-1] / df.iloc[df.index.get_loc(x.index[0]) - 1]['c'] - 1) * 100 if df.index.get_loc(x.index[0]) > 0 else np.nan)
    valid_blocks = condition.groupby(blocks).first()
    return durations[valid_blocks].values, changes[valid_blocks].values

  up_dur, up_chg = get_streaks(is_up)
  down_dur, down_chg = get_streaks(is_down)

  return up_dur, up_chg, down_dur, down_chg


def calculate_ordered_ma_trend_stats(df, periods):
  """
  Calculate duration and percentage change of trend following the 5 MA
  when the MAs are ordered (e.g., 5 MA > 10 MA > 20 MA) to the first close below the 5 MA.
  """
  # Use pre-calculated indicators from swing_indicators
  ma5 = df['ma5']
  
  # Condition for entry: all MAs are ordered
  # e.g., for [5, 10, 20]: (ma5 > ma10) & (ma10 > ma20)
  # e.g., for [5, 10]: (ma5 > ma10)
  
  long_condition = pd.Series(True, index=df.index)
  short_condition = pd.Series(True, index=df.index)
  
  for i in range(len(periods) - 1):
    ma_high = df[f'ma{periods[i]}']
    ma_low = df[f'ma{periods[i+1]}']
    long_condition &= (ma_high > ma_low)
    short_condition &= (ma_high < ma_low)

  def get_trend_stats(condition, exit_condition_func):
    # condition is the entry/persistence condition
    # exit_condition_func is a function of df and ma5 that returns true when we should exit
    # "to the first close below the 5 MA"
    
    in_trend = False
    start_idx = None
    results = []

    for i in range(len(df)):
      if not in_trend:
        if condition.iloc[i]:
          in_trend = True
          start_idx = i
      else:
        # Check for exit
        if exit_condition_func(df, ma5, i):
          duration = i - start_idx + 1
          # Use prior close of start_idx to include the entry bar move
          entry_price = df['c'].iloc[start_idx-1] if start_idx > 0 else df['c'].iloc[start_idx]
          change = (df['c'].iloc[i] / entry_price - 1) * 100
          results.append((duration, change))
          in_trend = False
    
    if results:
      durs, chgs = zip(*results)
      return np.array(durs), np.array(chgs)
    return np.array([]), np.array([])

  long_dur, long_chg = get_trend_stats(long_condition, lambda d, m, i: d['c'].iloc[i] < m.iloc[i])
  short_dur, short_chg = get_trend_stats(short_condition, lambda d, m, i: d['c'].iloc[i] > m.iloc[i])

  return long_dur, long_chg, short_dur, short_chg


## %%
# Prepare data
up_p_day, down_p_day = calculate_streak_probabilities(data.df_day, 'Daily')
up_p_week, down_p_week = calculate_streak_probabilities(data.df_week, 'Weekly')
up_p_month, down_p_month = calculate_streak_probabilities(data.df_month, 'Monthly')

ma_slope_stats_day = calculate_ma_slope_stats(data.df_day)
ma_slope_stats_week = calculate_ma_slope_stats(data.df_week)

hl_streak_dur_up, hl_streak_chg_up, hl_streak_dur_down, hl_streak_chg_down = calculate_high_low_streak_stats(data.df_week)

# Weekly 2-MA: 5 > 10
ma2_long_dur_w, ma2_long_chg_w, ma2_short_dur_w, ma2_short_chg_w = calculate_ordered_ma_trend_stats(data.df_week, [5, 10])

# Daily 3-MA: 5 > 10 > 20
ma3_long_dur_d, ma3_long_chg_d, ma3_short_dur_d, ma3_short_chg_d = calculate_ordered_ma_trend_stats(data.df_day, [5, 10, 20])

## %%
# Visualizations
plt.style.use('dark_background')
fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 2)  # Use 3 rows, 2 columns as base
# Row 1 will need special handling to fit 3 charts into 2 columns (or 6 columns total)

# Let's use 6 columns to allow for 3 items (2 cols each) and 2 items (3 cols each)
gs = fig.add_gridspec(3, 6)

# 1. Streak Probabilities (Daily, Weekly, Monthly)
ax1 = fig.add_subplot(gs[0, 0:2])
x = np.arange(2, 6)
ax1.bar(x - 0.25, [up_p_day[i] for i in x], 0.2, label='Up Day', color='green', alpha=0.7)
ax1.bar(x - 0.05, [up_p_week[i] for i in x], 0.2, label='Up Week', color='lightgreen', alpha=0.7)
ax1.bar(x + 0.15, [up_p_month[i] for i in x], 0.2, label='Up Month', color='darkgreen', alpha=0.7)

ax1.bar(x - 0.25, [-down_p_day[i] for i in x], 0.2, label='Down Day', color='red', alpha=0.7)
ax1.bar(x - 0.05, [-down_p_week[i] for i in x], 0.2, label='Down Week', color='salmon', alpha=0.7)
ax1.bar(x + 0.15, [-down_p_month[i] for i in x], 0.2, label='Down Month', color='darkred', alpha=0.7)

ax1.set_title('Prob. of Streak Length >= N')
ax1.set_xticks(x)
ax1.set_ylabel('Probability')
ax1.legend(fontsize='x-small', ncol=2)
ax1.axhline(0, color='white', linewidth=0.8)


# 2. Consecutive High/Low and Ordered MA Durations
ax2 = fig.add_subplot(gs[0, 2:4])
dur_data_combined = [
    hl_streak_dur_up,
    hl_streak_dur_down,
    ma2_long_dur_w,
    ma2_short_dur_w,
    ma3_long_dur_d,
    ma3_short_dur_d
]
dur_labels = ['H/L Up', 'H/L Down', 'W MA2 L', 'W MA2 S', 'D MA3 L', 'D MA3 S']
parts2 = ax2.violinplot(dur_data_combined, showmedians=False, showextrema=True, quantiles=[[0.25, 0.5, 0.75]] * len(dur_data_combined))
ax2.set_xticks(np.arange(1, len(dur_labels) + 1))
ax2.set_xticklabels(dur_labels, fontsize='x-small')
ax2.set_title('Weekly/Daily Trend Durations')
ax2.set_ylabel('Bars (W/D)')
annotate_violin(ax2, dur_data_combined, np.arange(1, len(dur_labels) + 1), dur_labels)

# 3. Consecutive High/Low and Ordered MA Price Changes
ax3 = fig.add_subplot(gs[0, 4:6])
chg_data_combined = [
    hl_streak_chg_up,
    hl_streak_chg_down,
    ma2_long_chg_w,
    ma2_short_chg_w,
    ma3_long_chg_d,
    ma3_short_chg_d
]
parts3 = ax3.violinplot(chg_data_combined, showmedians=False, showextrema=True, quantiles=[[0.25, 0.5, 0.75]] * len(chg_data_combined))
ax3.set_xticks(np.arange(1, len(dur_labels) + 1))
ax3.set_xticklabels(dur_labels, fontsize='x-small')
ax3.set_title('Weekly/Daily Trend Price Changes %')
ax3.set_ylabel('Change %')
annotate_violin(ax3, chg_data_combined, np.arange(1, len(dur_labels) + 1), dur_labels)

# 4. MA Slope Duration Violin Plot (Daily and Weekly)
ma_periods = [5, 10]

# Daily Duration
ax4_d = fig.add_subplot(gs[1, 0:3])
dur_data_d = []
labels_d = []
for p in ma_periods:
    dur_data_d.append(ma_slope_stats_day[p]['pos_dur'])
    labels_d.append(f'D MA{p}+')
    dur_data_d.append(ma_slope_stats_day[p]['neg_dur'])
    labels_d.append(f'D MA{p}-')

parts4d = ax4_d.violinplot(dur_data_d, showmedians=False, showextrema=True, quantiles=[[0.25, 0.5, 0.75]] * len(dur_data_d))
ax4_d.set_xticks(np.arange(1, len(labels_d) + 1))
ax4_d.set_xticklabels(labels_d, rotation=45, fontsize='x-small')
ax4_d.set_title('Daily MA Slope Duration (Days)')
ax4_d.set_ylabel('Days')
annotate_violin(ax4_d, dur_data_d, np.arange(1, len(labels_d) + 1), labels_d)

# Weekly Duration (Converted to Days)
ax4_w = fig.add_subplot(gs[1, 3:6])
dur_data_w = []
labels_w = []
for p in ma_periods:
    dur_data_w.append(ma_slope_stats_week[p]['pos_dur'] * 5)
    labels_w.append(f'W MA{p}+')
    dur_data_w.append(ma_slope_stats_week[p]['neg_dur'] * 5)
    labels_w.append(f'W MA{p}-')

parts4w = ax4_w.violinplot(dur_data_w, showmedians=False, showextrema=True, quantiles=[[0.25, 0.5, 0.75]] * len(dur_data_w))
ax4_w.set_xticks(np.arange(1, len(labels_w) + 1))
ax4_w.set_xticklabels(labels_w, rotation=45, fontsize='x-small')
ax4_w.set_title('Weekly MA Slope Duration (Days)')
ax4_w.set_ylabel('Days')
annotate_violin(ax4_w, dur_data_w, np.arange(1, len(labels_w) + 1), labels_w)

# 5. MA Slope Price Change Violin Plot (Daily and Weekly)
# Daily Price Change
ax5_d = fig.add_subplot(gs[2, 0:3])
chg_data_d = []
for p in ma_periods:
    chg_data_d.append(ma_slope_stats_day[p]['pos_chg'])
    chg_data_d.append(ma_slope_stats_day[p]['neg_chg'])

parts5d = ax5_d.violinplot(chg_data_d, showmedians=False, showextrema=True, quantiles=[[0.25, 0.5, 0.75]] * len(chg_data_d))
ax5_d.set_xticks(np.arange(1, len(labels_d) + 1))
ax5_d.set_xticklabels(labels_d, rotation=45, fontsize='x-small')
ax5_d.set_title('Daily MA Slope Price Change %')
ax5_d.set_ylabel('Change %')
annotate_violin(ax5_d, chg_data_d, np.arange(1, len(labels_d) + 1), labels_d)

# Weekly Price Change
ax5_w = fig.add_subplot(gs[2, 3:6])
chg_data_w = []
for p in ma_periods:
    chg_data_w.append(ma_slope_stats_week[p]['pos_chg'])
    chg_data_w.append(ma_slope_stats_week[p]['neg_chg'])

parts5w = ax5_w.violinplot(chg_data_w, showmedians=False, showextrema=True, quantiles=[[0.25, 0.5, 0.75]] * len(chg_data_w))
ax5_w.set_xticks(np.arange(1, len(labels_w) + 1))
ax5_w.set_xticklabels(labels_w, rotation=45, fontsize='x-small')
ax5_w.set_title('Weekly MA Slope Price Change %')
ax5_w.set_ylabel('Change %')
annotate_violin(ax5_w, chg_data_w, np.arange(1, len(labels_w) + 1), labels_w)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.show()


# %%
# =============================================================================
# MA20 Regime Filter Evaluation
# =============================================================================
# Question: When SPY is below MA20 but above MA50, does price typically recover
# (pullback within uptrend) or continue lower (real trend break)?
# This validates whether MA20 is a useful regime filter or too noisy.
#
# Three states classified per trading day:
#   Uptrend  — close > MA20
#   Pullback — close < MA20 AND close > MA50  (filter would block entry)
#   Breakdown— close < MA20 AND close < MA50  (genuine trend deterioration)
#
# Output per state:
#   - Episode count and median duration
#   - Max depth below MA20 (%)
#   - Forward returns at 5, 10, 20 days
#   - Recovery rate: % of Pullback episodes that close above MA20 within 10 days
# =============================================================================

# calculate_regime_filter_stats imported from finance.utils.move_character



# %%
# Load SPY for the regime filter evaluation (independent of the symbol above)
spy_data = utils.SwingTradingData('SPY', datasource='offline')
spy_df = spy_data.df_day

episodes_df, regime_summary = calculate_regime_filter_stats(spy_df)

# Print summary table
print(f"\n{'=' * 60}")
print(f"MA20 Regime Filter Evaluation — SPY")
print(f"{'=' * 60}")
for state, stats in regime_summary.items():
    if not stats:
        continue
    print(f"\n  [{state}]")
    for k, v in stats.items():
        print(f"    {k:<30} {v:.1f}")

pullback_eps = episodes_df[episodes_df['state'] == 'Pullback']
breakdown_eps = episodes_df[episodes_df['state'] == 'Breakdown']
uptrend_eps = episodes_df[episodes_df['state'] == 'Uptrend']

# %%
# Visualize regime filter evaluation
plt.style.use('dark_background')
fig2 = plt.figure(figsize=(24, 18))
fig2.suptitle(f'MA20 Regime Filter Evaluation — SPY', fontsize=14, color='white', y=0.98)
gs2 = fig2.add_gridspec(3, 6, hspace=0.45, wspace=0.35)

state_colors = {'Uptrend': '#24ad54', 'Pullback': '#f5a623', 'Breakdown': '#ec4533'}
forward_windows = [5, 10, 20]

# --- Row 1, left: Episode count bar chart ---
ax_count = fig2.add_subplot(gs2[0, 0:2])
states = ['Uptrend', 'Pullback', 'Breakdown']
counts = [regime_summary[s].get('n_episodes', 0) for s in states]
bars = ax_count.bar(states, counts, color=[state_colors[s] for s in states], alpha=0.8)
for bar, val in zip(bars, counts):
    ax_count.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  str(int(val)), ha='center', va='bottom', color='white', fontsize='small')
ax_count.set_title('Episode Count by State')
ax_count.set_ylabel('# Episodes')

# --- Row 1, centre: Recovery rate (Pullback only) ---
ax_recov = fig2.add_subplot(gs2[0, 2:4])
recovery_rate = regime_summary['Pullback'].get('recovery_rate_pct', 0)
ax_recov.bar(['Recovery\n(≤10 days)', 'No Recovery'], [recovery_rate, 100 - recovery_rate],
             color=['#24ad54', '#ec4533'], alpha=0.8)
ax_recov.set_ylim(0, 100)
ax_recov.set_title(f'Pullback Recovery Rate\n(close > MA20 within 10 days)')
ax_recov.set_ylabel('% of Pullback Episodes')
ax_recov.text(0, recovery_rate + 1, f'{recovery_rate:.1f}%', ha='center', color='white', fontsize='small')
ax_recov.text(1, (100 - recovery_rate) + 1, f'{100 - recovery_rate:.1f}%', ha='center', color='white', fontsize='small')

# --- Row 1, right: Median duration by state ---
ax_dur_bar = fig2.add_subplot(gs2[0, 4:6])
med_durs = [regime_summary[s].get('med_duration', 0) for s in states]
bars2 = ax_dur_bar.bar(states, med_durs, color=[state_colors[s] for s in states], alpha=0.8)
for bar, val in zip(bars2, med_durs):
    ax_dur_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}d', ha='center', va='bottom', color='white', fontsize='small')
ax_dur_bar.set_title('Median Episode Duration (Days)')
ax_dur_bar.set_ylabel('Days')

# --- Row 2: Duration violin plots (Pullback vs Breakdown) ---
ax_dur_v = fig2.add_subplot(gs2[1, 0:3])
dur_violin_data = [
    pullback_eps['duration'].dropna().values,
    breakdown_eps['duration'].dropna().values,
]
dur_violin_labels = ['Pullback', 'Breakdown']
valid_dur = [d for d in dur_violin_data if len(d) > 1]
valid_dur_labels = [l for d, l in zip(dur_violin_data, dur_violin_labels) if len(d) > 1]
if valid_dur:
    parts = ax_dur_v.violinplot(valid_dur, showmedians=False, showextrema=True,
                                 quantiles=[[0.25, 0.5, 0.75]] * len(valid_dur))
    for pc, state in zip(parts['bodies'], valid_dur_labels):
        pc.set_facecolor(state_colors[state])
        pc.set_alpha(0.6)
    ax_dur_v.set_xticks(np.arange(1, len(valid_dur_labels) + 1))
    ax_dur_v.set_xticklabels(valid_dur_labels)
    annotate_violin(ax_dur_v, valid_dur, np.arange(1, len(valid_dur_labels) + 1), valid_dur_labels)
ax_dur_v.set_title('Episode Duration Distribution (Days)')
ax_dur_v.set_ylabel('Days')

# --- Row 2: Max depth violin plots ---
ax_depth_v = fig2.add_subplot(gs2[1, 3:6])
depth_violin_data = [
    pullback_eps['max_depth_pct'].dropna().values,
    breakdown_eps['max_depth_pct'].dropna().values,
]
valid_depth = [d for d in depth_violin_data if len(d) > 1]
valid_depth_labels = [l for d, l in zip(depth_violin_data, dur_violin_labels) if len(d) > 1]
if valid_depth:
    parts2 = ax_depth_v.violinplot(valid_depth, showmedians=False, showextrema=True,
                                    quantiles=[[0.25, 0.5, 0.75]] * len(valid_depth))
    for pc, state in zip(parts2['bodies'], valid_depth_labels):
        pc.set_facecolor(state_colors[state])
        pc.set_alpha(0.6)
    ax_depth_v.set_xticks(np.arange(1, len(valid_depth_labels) + 1))
    ax_depth_v.set_xticklabels(valid_depth_labels)
    annotate_violin(ax_depth_v, valid_depth, np.arange(1, len(valid_depth_labels) + 1), valid_depth_labels)
ax_depth_v.axhline(0, color='#666', linewidth=0.8, linestyle='--')
ax_depth_v.set_title('Max Depth Below MA20 (%)')
ax_depth_v.set_ylabel('% Below MA20')

# --- Row 3: Forward returns by state (5d, 10d, 20d) ---
fwd_col_pairs = [('fwd_5', '5d'), ('fwd_10', '10d'), ('fwd_20', '20d')]
for col_idx, (fwd_col, fwd_label) in enumerate(fwd_col_pairs):
    ax_fwd = fig2.add_subplot(gs2[2, col_idx * 2: col_idx * 2 + 2])
    fwd_violin_data = []
    fwd_violin_labels = []
    for state, ep_df in [('Uptrend', uptrend_eps), ('Pullback', pullback_eps), ('Breakdown', breakdown_eps)]:
        if fwd_col in ep_df.columns:
            vals = ep_df[fwd_col].dropna().values
            if len(vals) > 1:
                fwd_violin_data.append(vals)
                fwd_violin_labels.append(state)

    if fwd_violin_data:
        parts3 = ax_fwd.violinplot(fwd_violin_data, showmedians=False, showextrema=True,
                                    quantiles=[[0.25, 0.5, 0.75]] * len(fwd_violin_data))
        for pc, state in zip(parts3['bodies'], fwd_violin_labels):
            pc.set_facecolor(state_colors[state])
            pc.set_alpha(0.6)
        ax_fwd.set_xticks(np.arange(1, len(fwd_violin_labels) + 1))
        ax_fwd.set_xticklabels(fwd_violin_labels, fontsize='x-small')
        annotate_violin(ax_fwd, fwd_violin_data, np.arange(1, len(fwd_violin_labels) + 1), fwd_violin_labels)

    ax_fwd.axhline(0, color='#666', linewidth=0.8, linestyle='--')
    ax_fwd.set_title(f'Forward Return — {fwd_label}')
    ax_fwd.set_ylabel('Return %')

plt.show()


# %%
# =============================================================================
# Block 1 — Move Magnitude Distribution
# =============================================================================
# Are daily moves explosive (fat-tailed) or gradual?
# Determines: far OTM vs ATM credit spread suitability.
#
# Metric  : daily return normalized by ATR20 (instrument-agnostic scale)
# Tails   : % of days exceeding 1×, 1.5×, 2×, 3× ATR — up and down separately
# Flag    : EXPLOSIVE if >2× ATR moves occur on >5% of days
# QQ plot : deviation from normal distribution — fat tails visible as S-curve divergence
# =============================================================================

# calculate_move_magnitude_stats imported from finance.utils.move_character


# %%
# Compute — Block 1
norm_moves, tail_freq, excess_kurtosis, skewness = calculate_move_magnitude_stats(data.df_day)

flag_move = 'EXPLOSIVE' if tail_freq[2.0]['total'] > 5.0 else 'GRADUAL'
print(f'\n{"=" * 55}')
print(f'Move Character: {flag_move}  |  Symbol: {symbol}')
print(f'  Excess kurtosis : {excess_kurtosis:.2f}  (normal = 0)')
print(f'  Skewness        : {skewness:.2f}  (normal = 0)')
print(f'  {"Threshold":<14} {"Up %":>6}  {"Down %":>7}  {"Total %":>8}')
for t, v in tail_freq.items():
    print(f'  |move| > {t:.1f}×ATR  {v["up"]:>6.1f}  {v["down"]:>7.1f}  {v["total"]:>8.1f}')

# %%
# Visualize — Block 1
from scipy.stats import norm as _scipy_norm

plt.style.use('dark_background')
fig3 = plt.figure(figsize=(24, 8))
fig3.suptitle(f'Move Magnitude Distribution — {symbol}  [{flag_move}]', fontsize=13, color='white')
gs3 = fig3.add_gridspec(1, 3, wspace=0.35)

# --- Left: Histogram of ATR-normalized moves with tail threshold lines ---
ax3a = fig3.add_subplot(gs3[0, 0])
vals3 = norm_moves.dropna().values
lo3, hi3 = np.percentile(vals3, 0.5), np.percentile(vals3, 99.5)
ax3a.hist(vals3, bins=np.linspace(lo3, hi3, 60), color='#48dbfb', alpha=0.7, density=True)
ax3a.set_title('ATR20-Normalized Daily Moves')
ax3a.set_xlabel('Move / ATR20')
ax3a.set_ylabel('Density')
for t, color in [(1.0, '#f5a623'), (1.5, '#ff8c00'), (2.0, '#ec4533'), (3.0, '#9b59b6')]:
    freq = tail_freq[t]['total']
    ax3a.axvline( t, color=color, linestyle='--', linewidth=1.2, label=f'>{t:.1f}× ({freq:.1f}%)')
    ax3a.axvline(-t, color=color, linestyle='--', linewidth=1.2)
ax3a.legend(fontsize='x-small')
ax3a.text(0.02, 0.97, f'Kurt: {excess_kurtosis:.1f}  Skew: {skewness:.2f}',
          transform=ax3a.transAxes, va='top', fontsize='x-small', color='#aaa')

# --- Centre: Tail frequency bar chart — up vs down at each threshold ---
ax3b = fig3.add_subplot(gs3[0, 1])
thresh_labels = [f'>{t:.1f}×' for t in tail_freq]
up_vals   = [tail_freq[t]['up']   for t in tail_freq]
down_vals = [tail_freq[t]['down'] for t in tail_freq]
x3 = np.arange(len(thresh_labels))
w3 = 0.35
ax3b.bar(x3 - w3 / 2, up_vals,   w3, label='Up',   color='#24ad54', alpha=0.8)
ax3b.bar(x3 + w3 / 2, down_vals, w3, label='Down', color='#ec4533', alpha=0.8)
for i, (u, d) in enumerate(zip(up_vals, down_vals)):
    ax3b.text(i - w3 / 2, u + 0.05, f'{u:.1f}', ha='center', fontsize='xx-small', color='white')
    ax3b.text(i + w3 / 2, d + 0.05, f'{d:.1f}', ha='center', fontsize='xx-small', color='white')
ax3b.set_xticks(x3)
ax3b.set_xticklabels(thresh_labels)
ax3b.set_title('Tail Frequencies by Direction (%)')
ax3b.set_ylabel('% of Days')
ax3b.legend(fontsize='x-small')
ax3b.axhline(5.0, color='#f5a623', linestyle=':', linewidth=1, alpha=0.7)
ax3b.text(len(thresh_labels) - 0.5, 5.3, 'explosive threshold',
          ha='right', fontsize='xx-small', color='#f5a623')

# --- Right: QQ plot — empirical standardized returns vs theoretical normal ---
ax3c = fig3.add_subplot(gs3[0, 2])
pct_raw = data.df_day['pct'].dropna().values
mean_v, std_v = pct_raw.mean(), pct_raw.std(ddof=1)
std_sorted = np.sort((pct_raw - mean_v) / std_v)
n_qq = len(std_sorted)
theoretical_q = _scipy_norm.ppf((np.arange(1, n_qq + 1) - 0.5) / n_qq)
ax3c.scatter(theoretical_q, std_sorted, s=3, alpha=0.4, color='#48dbfb')
ref = np.array([theoretical_q[0], theoretical_q[-1]])
ax3c.plot(ref, ref, color='#f5a623', linewidth=1.5, linestyle='--', label='Normal')
ax3c.set_title('QQ Plot — Returns vs Normal')
ax3c.set_xlabel('Theoretical Quantiles')
ax3c.set_ylabel('Empirical Quantiles')
ax3c.legend(fontsize='x-small')
ax3c.text(0.02, 0.97, f'Excess kurtosis: {excess_kurtosis:.2f}',
          transform=ax3c.transAxes, va='top', fontsize='x-small', color='#aaa')

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Block 2 — Intra-Trend Retracement Depth
# =============================================================================
# Within a confirmed trend (MAs ordered), how far does price pull back from the
# running peak before the trend resumes or ends?
#
# Determines: minimum viable stop distance to stay in a winning position.
# Q75 of intra-trend retracement = practical stop floor.
# Scatter of duration vs retracement shows whether longer trends are noisier.
# =============================================================================

# calculate_intratrend_retracement imported from finance.utils.move_character


# %%
# Compute — Block 2
long_retr, short_retr = calculate_intratrend_retracement(data.df_day)
med_atr20 = data.df_day['atrp20'].median()

print(f'\nIntra-Trend Retracement — {symbol}')
for label, df_r in [('Long (MA5>MA10>MA20)', long_retr), ('Short (MA5<MA10<MA20)', short_retr)]:
    if df_r.empty:
        print(f'  {label}: no episodes found')
        continue
    vals_r = df_r['max_retracement_pct'].abs()
    q25r, q50r, q75r = np.percentile(vals_r, [25, 50, 75])
    print(f'  {label}:')
    print(f'    Episodes : {len(df_r)}')
    print(f'    Q25/50/75: {q25r:.2f}% / {q50r:.2f}% / {q75r:.2f}%')
    print(f'    Q50 vs ATR20: {q50r / med_atr20:.2f}×  |  Q75 vs ATR20: {q75r / med_atr20:.2f}×')

# %%
# Visualize — Block 2
plt.style.use('dark_background')
fig4 = plt.figure(figsize=(24, 9))
fig4.suptitle(f'Intra-Trend Retracement Depth — {symbol}', fontsize=13, color='white')
gs4 = fig4.add_gridspec(1, 2, wspace=0.35)

# --- Left: Violin of max retracement magnitude (long vs short) ---
ax4a = fig4.add_subplot(gs4[0, 0])
vdata4, vlabels4 = [], []
dir_colors4 = {'Long': '#24ad54', 'Short': '#ec4533'}
for label, df_r in [('Long', long_retr), ('Short', short_retr)]:
    if not df_r.empty:
        vals = df_r['max_retracement_pct'].abs().dropna().values
        if len(vals) > 1:
            vdata4.append(vals)
            vlabels4.append(label)

if vdata4:
    parts4 = ax4a.violinplot(vdata4, showmedians=False, showextrema=True,
                              quantiles=[[0.25, 0.5, 0.75]] * len(vdata4))
    for pc, lbl in zip(parts4['bodies'], vlabels4):
        pc.set_facecolor(dir_colors4[lbl])
        pc.set_alpha(0.6)
    ax4a.set_xticks(np.arange(1, len(vlabels4) + 1))
    ax4a.set_xticklabels(vlabels4)
    annotate_violin(ax4a, vdata4, np.arange(1, len(vlabels4) + 1), vlabels4)

for mult, color, ls in [(0.5, '#f5a623', ':'), (1.0, '#ff6b6b', '--'), (1.5, '#ec4533', '--')]:
    ref = med_atr20 * mult
    ax4a.axhline(ref, color=color, linestyle=ls, linewidth=1.2,
                 label=f'{mult:.1f}× ATR20 ({ref:.1f}%)')
ax4a.set_title('Max Intra-Trend Retracement (Abs %)')
ax4a.set_ylabel('Retracement %')
ax4a.legend(fontsize='x-small')

# --- Right: Scatter — duration vs retracement ---
ax4b = fig4.add_subplot(gs4[0, 1])
for label, df_r, color in [('Long', long_retr, '#24ad54'), ('Short', short_retr, '#ec4533')]:
    if not df_r.empty:
        ax4b.scatter(df_r['duration'], df_r['max_retracement_pct'].abs(),
                     alpha=0.4, s=15, color=color, label=label)
for mult, color in [(0.5, '#f5a623'), (1.0, '#ff6b6b'), (1.5, '#ec4533')]:
    ax4b.axhline(med_atr20 * mult, color=color, linestyle=':', linewidth=0.8)
ax4b.set_title('Trend Duration vs Max Retracement')
ax4b.set_xlabel('Duration (Days)')
ax4b.set_ylabel('Max Retracement %')
ax4b.legend(fontsize='x-small')

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Block 3 — Gap vs Intraday Range Decomposition
# =============================================================================
# What fraction of daily true range comes from overnight gaps vs intraday moves?
#
# Overnight gaps are unhedgeable with stop-losses and change short premium risk
# regardless of IVR. High gap contribution means directional stops will not
# protect against the most damaging moves.
#
# Flag: GAP-HEAVY if mean gap contribution > 40% of true range.
# =============================================================================

# calculate_gap_intraday_decomposition imported from finance.utils.move_character


# %%
# Compute — Block 3
gap_stats, rolling_gap = calculate_gap_intraday_decomposition(data.df_day)

mean_gap_contrib = gap_stats['gap_contrib'].mean() * 100
fill_rate = gap_stats.loc[gap_stats['gap_dir'] != 0, 'gap_filled'].mean() * 100
flag_gap  = 'GAP-HEAVY' if mean_gap_contrib > 40 else 'INTRADAY-DRIVEN'

print(f'\nGap Decomposition — {symbol}  [{flag_gap}]')
print(f'  Mean gap contribution  : {mean_gap_contrib:.1f}% of true range')
print(f'  Gap fill rate (same day): {fill_rate:.1f}%')
print(f'  Mean overnight gap     : {gap_stats["overnight_gap_pct"].abs().mean():.2f}%')
print(f'  Mean intraday range    : {gap_stats["intraday_range_pct"].mean():.2f}%')
print(f'  Mean true range (ATR1) : {gap_stats["true_range_pct"].mean():.2f}%')

# %%
# Visualize — Block 3
plt.style.use('dark_background')
fig5 = plt.figure(figsize=(24, 12))
fig5.suptitle(f'Gap vs Intraday Range Decomposition — {symbol}  [{flag_gap}]',
              fontsize=13, color='white')
gs5 = fig5.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

# --- Top-left: Histogram of gap contribution ratio ---
ax5a = fig5.add_subplot(gs5[0, 0])
ax5a.hist(gap_stats['gap_contrib'].dropna(), bins=40, color='#48dbfb', alpha=0.7, density=True)
ax5a.axvline(0.4, color='#f5a623', linestyle='--', linewidth=1.5, label='Gap-heavy threshold (40%)')
ax5a.axvline(gap_stats['gap_contrib'].mean(), color='white', linestyle=':', linewidth=1.2,
             label=f'Mean: {gap_stats["gap_contrib"].mean():.2f}')
ax5a.set_title('Gap Contribution to True Range')
ax5a.set_xlabel('Gap / True Range  (0 = pure intraday, 1 = pure gap)')
ax5a.set_ylabel('Density')
ax5a.legend(fontsize='x-small')
pct_gap_dom = (gap_stats['gap_contrib'] > 0.5).mean() * 100
ax5a.text(0.97, 0.97, f'{pct_gap_dom:.1f}% of days gap-dominated',
          transform=ax5a.transAxes, ha='right', va='top', fontsize='x-small', color='#f5a623')

# --- Top-right: Scatter overnight gap vs intraday range, coloured by gap direction ---
ax5b = fig5.add_subplot(gs5[0, 1])
for mask, color, label in [
    (gap_stats['gap_dir'] > 0,  '#24ad54', 'Gap up'),
    (gap_stats['gap_dir'] < 0,  '#ec4533', 'Gap down'),
    (gap_stats['gap_dir'] == 0, '#aaaaaa', 'Flat open'),
]:
    ax5b.scatter(gap_stats.loc[mask, 'overnight_gap_pct'],
                 gap_stats.loc[mask, 'intraday_range_pct'],
                 s=5, alpha=0.3, color=color, label=label)
ax5b.set_title('Overnight Gap vs Intraday Range')
ax5b.set_xlabel('Overnight Gap %')
ax5b.set_ylabel('Intraday Range %')
ax5b.legend(fontsize='x-small', markerscale=3)
ax5b.text(0.02, 0.97, f'Fill rate: {fill_rate:.1f}%',
          transform=ax5b.transAxes, va='top', fontsize='x-small', color='#aaa')

# --- Bottom-left: Rolling 63-day gap contribution over time ---
ax5c = fig5.add_subplot(gs5[1, 0])
valid_roll = rolling_gap.dropna()
ax5c.plot(valid_roll.index, valid_roll.values * 100, color='#48dbfb', linewidth=1.2)
ax5c.axhline(40, color='#f5a623', linestyle='--', linewidth=1, label='40% threshold')
ax5c.fill_between(valid_roll.index, valid_roll.values * 100, 40,
                   where=(valid_roll.values * 100 > 40),
                   color='#f5a623', alpha=0.2, label='Gap-heavy periods')
ax5c.set_title('Rolling 63-Day Gap Contribution (%)')
ax5c.set_ylabel('Gap Contribution %')
ax5c.legend(fontsize='x-small')

# --- Bottom-right: Average component breakdown bar chart ---
ax5d = fig5.add_subplot(gs5[1, 1])
comp_labels = ['Overnight\nGap', 'Intraday\nRange', 'True Range\n(ATR1)']
comp_vals   = [
    gap_stats['overnight_gap_pct'].abs().mean(),
    gap_stats['intraday_range_pct'].mean(),
    gap_stats['true_range_pct'].mean(),
]
bars5 = ax5d.bar(comp_labels, comp_vals, color=['#f5a623', '#48dbfb', 'white'], alpha=0.8)
for bar, val in zip(bars5, comp_vals):
    ax5d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
              f'{val:.2f}%', ha='center', va='bottom', fontsize='small', color='white')
ax5d.set_title('Average Volatility Components (%)')
ax5d.set_ylabel('% of Price')

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Block 4 — Realized Volatility Regime Analysis
# =============================================================================
# Is HV stable or does it switch regimes unpredictably?
# Regime-switching vol punishes ATM credit sellers with sudden expansion.
#
# Regimes classified using rolling 252-day percentile thresholds (adaptive):
#   Low    — HV < 33rd percentile of trailing year
#   Medium — 33rd–67th percentile
#   High   — > 67th percentile
#
# Key metric: P(Low → High direct) — if >15%, vol spikes are non-gradual.
# Uses hvc (brokerage close HV) if available, falls back to hv20.
# =============================================================================

# calculate_hv_regime_stats imported from finance.utils.move_character


# %%
# Compute — Block 4
hv_series, hv_regime, hv_episodes, hv_transitions, hv_col_used = \
    calculate_hv_regime_stats(data.df_day)

flag_hv = 'N/A'  # default; overwritten below if HV data is available
if hv_series is not None:
    direct_spike = float(hv_transitions.loc['Low', 'High']) \
        if 'Low' in hv_transitions.index else 0.0
    flag_hv = 'SPIKE-PRONE' if direct_spike > 15 else 'MEAN-REVERTING'

    print(f'\nHV Regime Analysis — {symbol}  [{flag_hv}]  (col: {hv_col_used})')
    print(f'  P(Low → High direct spike) : {direct_spike:.1f}%')
    for reg in ['Low', 'Medium', 'High']:
        sub = hv_episodes[hv_episodes['regime'] == reg]
        if not sub.empty:
            print(f'  {reg:<8}: {len(sub):>3} episodes  |  '
                  f'median duration {sub["duration"].median():.0f} days')
    print('\n  Transition Matrix (%):')
    print(hv_transitions.to_string(float_format=lambda x: f'{x:.1f}'))

# %%
# Visualize — Block 4
if hv_series is not None:
    plt.style.use('dark_background')
    fig6 = plt.figure(figsize=(24, 12))
    fig6.suptitle(
        f'Realized Volatility Regime Analysis — {symbol}  [{flag_hv}]  ({hv_col_used})',
        fontsize=13, color='white')
    gs6 = fig6.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    regime_colors = {'Low': '#24ad54', 'Medium': '#f5a623', 'High': '#ec4533'}

    # --- Top row: HV time series with regime shading (full width) ---
    ax6a = fig6.add_subplot(gs6[0, :])
    ax6a.plot(hv_series.index, hv_series.values, color='white', linewidth=0.8, alpha=0.9)
    for reg, color in regime_colors.items():
        mask = hv_regime.reindex(hv_series.index, fill_value='Medium') == reg
        ax6a.fill_between(hv_series.index, 0, hv_series.values,
                           where=mask, color=color, alpha=0.25, label=reg)
    ax6a.set_title(f'{hv_col_used.upper()} with Regime Shading')
    ax6a.set_ylabel('Realized Volatility (%)')
    ax6a.legend(fontsize='x-small', loc='upper left')

    # --- Bottom-left: Violin of episode durations per regime ---
    ax6b = fig6.add_subplot(gs6[1, 0])
    vdata6, vlabels6 = [], []
    for reg in ['Low', 'Medium', 'High']:
        vals = hv_episodes[hv_episodes['regime'] == reg]['duration'].dropna().values
        if len(vals) > 1:
            vdata6.append(vals)
            vlabels6.append(reg)
    if vdata6:
        parts6 = ax6b.violinplot(vdata6, showmedians=False, showextrema=True,
                                  quantiles=[[0.25, 0.5, 0.75]] * len(vdata6))
        for pc, lbl in zip(parts6['bodies'], vlabels6):
            pc.set_facecolor(regime_colors[lbl])
            pc.set_alpha(0.6)
        ax6b.set_xticks(np.arange(1, len(vlabels6) + 1))
        ax6b.set_xticklabels(vlabels6)
        annotate_violin(ax6b, vdata6, np.arange(1, len(vlabels6) + 1), vlabels6)
    ax6b.set_title('Regime Episode Duration (Days)')
    ax6b.set_ylabel('Days')

    # --- Bottom-right: Transition probability heatmap ---
    ax6c = fig6.add_subplot(gs6[1, 1])
    mat6 = hv_transitions.values.astype(float)
    im6  = ax6c.imshow(mat6, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax6c.set_xticks([0, 1, 2])
    ax6c.set_yticks([0, 1, 2])
    ax6c.set_xticklabels(['Low', 'Medium', 'High'])
    ax6c.set_yticklabels(['Low', 'Medium', 'High'])
    ax6c.set_xlabel('To Regime')
    ax6c.set_ylabel('From Regime')
    ax6c.set_title('Regime Transition Probabilities (%)')
    for i in range(3):
        for j in range(3):
            val = mat6[i, j] if i < mat6.shape[0] and j < mat6.shape[1] else 0.0
            ax6c.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize='small',
                      fontweight='bold', color='black' if val > 50 else 'white')
    fig6.colorbar(im6, ax=ax6c, label='Probability (%)')

    plt.tight_layout()
    plt.show()
