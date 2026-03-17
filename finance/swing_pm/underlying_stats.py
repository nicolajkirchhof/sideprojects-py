# %%
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
# %load_ext autoreload
# %autoreload 2

# %%
symbol = 'QQQ'

data = utils.SwingTradingData(symbol, datasource='offline')


# df_dolt = utils.swing_trading_data.SwingTradingData(symbol)

#%%
# Questions on stock
# - probability of two, three, four, five successive down/up days/weeks/month
# - duration and percentage change following 5, 10, 20 MA daily, weekly. Following is defined as the slope of the MA becoming positive / negative until it turns into the opposite direction.
# - duration and percentage change of consecutive weekly up / down moves that are defined as the close not being below / above the prior high / low
# - duration and percentage change of weekly trend following the 5 MA when the 5 MA > 10 MA > 20 MA to the first close below the 5 MA. The same with opposite MAs for negative trend following

def calculate_streak_probabilities(df, label):
  """
  Calculate the probability of consecutive up/down streaks.
  """
  # streak is already in df from swing_indicators (via SwingTradingData)
  # or it can be recalculated to be sure
  change_sign = np.sign(df['pct']).replace(0, np.nan).ffill()
  streak_groups = (change_sign != change_sign.shift()).cumsum()
  streaks = change_sign.groupby(streak_groups).cumsum()

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


def calculate_ma_slope_stats(df, ma_periods=[5, 10, 20]):
  """
  Calculate duration and change following 5, 10, 20 MA.
  Following is defined as the slope of the MA becoming positive / negative until it turns into the opposite direction.
  """
  stats = {}
  for p in ma_periods:
    ma = df['c'].rolling(window=p).mean()
    slope = ma.diff()
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
    # For price change, we want the change from the start of the block to the end
    changes = group.apply(lambda x: (x['c'].iloc[-1] / x['c'].iloc[0] - 1) * 100)

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
  defined as the close not being below / above the prior high / low.
  """
  # For 'Up' move: close >= prior high
  # For 'Down' move: close <= prior low
  # This is a bit tricky because what happens if it's in between?
  # The requirement says "consecutive weekly up / down moves that are defined as the close not being below / above the prior high / low"
  # Let's interpret:
  # Up streak: close >= prior high. Continues as long as close >= prior high?
  # Actually, "close not being below prior high" might mean an Up streak continues as long as close >= prior high.
  # Let's use:
  # Up move: close >= prev high
  # Down move: close <= prev low

  prev_h = df['h'].shift()
  prev_l = df['l'].shift()

  is_up = df['c'] >= prev_h
  is_down = df['c'] <= prev_l

  # Since a bar can't be both >= prev_h and <= prev_l (usually), but it could be neither.
  # If it's neither, the streak breaks.
  # We need to identify streaks of 'is_up' and streaks of 'is_down'.

  def get_streaks(condition):
    blocks = (condition != condition.shift()).cumsum()
    group = df.groupby(blocks)
    durations = group.size()
    changes = group.apply(lambda x: (x['c'].iloc[-1] / x['c'].iloc[0] - 1) * 100)
    valid_blocks = condition.groupby(blocks).first()
    return durations[valid_blocks].values, changes[valid_blocks].values

  up_dur, up_chg = get_streaks(is_up)
  down_dur, down_chg = get_streaks(is_down)

  return up_dur, up_chg, down_dur, down_chg


def calculate_triple_ma_trend_stats(df):
  """
  Calculate duration and percentage change of weekly trend following the 5 MA
  when the 5 MA > 10 MA > 20 MA to the first close below the 5 MA.
  """
  ma5 = df['c'].rolling(window=5).mean()
  ma10 = df['c'].rolling(window=10).mean()
  ma20 = df['c'].rolling(window=20).mean()

  # Long trend: ma5 > ma10 > ma20
  # Short trend: ma5 < ma10 < ma20
  long_condition = (ma5 > ma10) & (ma10 > ma20)
  short_condition = (ma5 < ma10) & (ma10 < ma20)

  def get_trend_stats(condition, exit_condition_func):
    # condition is the entry/persistence condition (e.g., ma5 > ma10 > ma20)
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
          change = (df['c'].iloc[i] / df['c'].iloc[start_idx] - 1) * 100
          results.append((duration, change))
          in_trend = False
    
    if results:
      durs, chgs = zip(*results)
      return np.array(durs), np.array(chgs)
    return np.array([]), np.array([])

  long_dur, long_chg = get_trend_stats(long_condition, lambda d, m, i: d['c'].iloc[i] < m.iloc[i])
  short_dur, short_chg = get_trend_stats(short_condition, lambda d, m, i: d['c'].iloc[i] > m.iloc[i])

  return long_dur, long_chg, short_dur, short_chg


# %%
# Prepare data
up_p_day, down_p_day = calculate_streak_probabilities(data.df_day, 'Daily')
up_p_week, down_p_week = calculate_streak_probabilities(data.df_week, 'Weekly')
up_p_month, down_p_month = calculate_streak_probabilities(data.df_month, 'Monthly')

ma_slope_stats_day = calculate_ma_slope_stats(data.df_day)
ma_slope_stats_week = calculate_ma_slope_stats(data.df_week)

hl_streak_dur_up, hl_streak_chg_up, hl_streak_dur_down, hl_streak_chg_down = calculate_high_low_streak_stats(data.df_week)

triple_ma_long_dur, triple_ma_long_chg, triple_ma_short_dur, triple_ma_short_chg = calculate_triple_ma_trend_stats(data.df_week)

# %%
# Visualizations
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3)

# 1. Streak Probabilities (Daily)
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(2, 6)
ax1.bar(x - 0.2, [up_p_day[i] for i in x], 0.4, label='Up Streaks', color='green', alpha=0.7)
ax1.bar(x + 0.2, [down_p_day[i] for i in x], 0.4, label='Down Streaks', color='red', alpha=0.7)
ax1.set_title('Prob. of Streak Length >= N (Daily)')
ax1.set_xticks(x)
ax1.set_ylabel('Probability')
ax1.legend()

# 2. MA Slope Duration (Daily)
ax2 = fig.add_subplot(gs[0, 1])
ma_to_plot = 10
ax2.hist(ma_slope_stats_day[ma_to_plot]['pos_dur'], bins=30, alpha=0.5, label=f'Pos Slope MA{ma_to_plot}', color='green')
ax2.hist(ma_slope_stats_day[ma_to_plot]['neg_dur'], bins=30, alpha=0.5, label=f'Neg Slope MA{ma_to_plot}', color='red')
ax2.set_title(f'Duration Following {ma_to_plot} MA Slope (Daily)')
ax2.set_xlabel('Days')
ax2.legend()

# 3. MA Slope Change (Daily)
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(ma_slope_stats_day[ma_to_plot]['pos_chg'], bins=30, alpha=0.5, label=f'Pos Slope MA{ma_to_plot}', color='green')
ax3.hist(ma_slope_stats_day[ma_to_plot]['neg_chg'], bins=30, alpha=0.5, label=f'Neg Slope MA{ma_to_plot}', color='red')
ax3.set_title(f'Price Change % Following {ma_to_plot} MA Slope (Daily)')
ax3.set_xlabel('Change %')
ax3.legend()

# 4. Weekly Close vs Prior High/Low Streak Duration
ax4 = fig.add_subplot(gs[1, 0])
ax4.boxplot([hl_streak_dur_up, hl_streak_dur_down], tick_labels=['C >= Prev H', 'C <= Prev L'])
ax4.set_title('Duration of Weekly High/Low Streaks')
ax4.set_ylabel('Weeks')

# 5. Weekly Close vs Prior High/Low Streak Change %
ax5 = fig.add_subplot(gs[1, 1])
ax5.boxplot([hl_streak_chg_up, hl_streak_chg_down], tick_labels=['C >= Prev H', 'C <= Prev L'])
ax5.set_title('Price Change % of Weekly High/Low Streaks')
ax5.set_ylabel('Change %')

# 6. Triple MA Trend Duration (Weekly)
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(triple_ma_long_dur, bins=15, alpha=0.5, label='5>10>20 Trend', color='green')
ax6.hist(triple_ma_short_dur, bins=15, alpha=0.5, label='5<10<20 Trend', color='red')
ax6.set_title('Triple MA Trend Duration (Weekly)')
ax6.set_xlabel('Weeks')
ax6.legend()

# 7. Triple MA Trend Change % (Weekly)
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(triple_ma_long_chg, bins=15, alpha=0.5, label='5>10>20 Trend', color='green')
ax7.hist(triple_ma_short_chg, bins=15, alpha=0.5, label='5<10<20 Trend', color='red')
ax7.set_title('Triple MA Trend Change % (Weekly)')
ax7.set_xlabel('Change %')
ax7.legend()

# 8. Average Change by MA Slope (Summary bar chart)
ax8 = fig.add_subplot(gs[2, 1:])
ma_labels = ['MA5', 'MA10', 'MA20']
avg_pos_chg = [ma_slope_stats_day[p]['pos_chg'].mean() for p in [5, 10, 20]]
avg_neg_chg = [ma_slope_stats_day[p]['neg_chg'].mean() for p in [5, 10, 20]]

ax8.bar(np.arange(3) - 0.2, avg_pos_chg, 0.4, label='Avg Change Pos Slope', color='green', alpha=0.7)
ax8.bar(np.arange(3) + 0.2, avg_neg_chg, 0.4, label='Avg Change Neg Slope', color='red', alpha=0.7)
ax8.set_title('Avg Price Change % Following MA Slope (Daily)')
ax8.set_xticks(np.arange(3))
ax8.set_xticklabels(ma_labels)
ax8.set_ylabel('Avg Change %')
ax8.legend()

plt.tight_layout()
plt.show()

