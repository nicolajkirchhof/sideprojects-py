# %%
import datetime
import glob

import dateutil
import numpy as np
import scipy

import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
import pickle
import numpy as np
import humanize
import glob


from matplotlib.pyplot import tight_layout

import finance.utils as utils

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2
#%%
# TODO:
# - Determine time between Significant points
# - How does it change in comparison to the last hour before the opening. Is that a good setup?
# - Analyze the changes thoughout the day, how many percentages are made where
#   - What might be triggers for a positive/negative rally into the close of the US indices?
# -
# Trends & Patterns
# - Follow-Through on strong trend without pullback vs 50% PB vs 100% PB on the same day


# %%
symbols = ['IBDE40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225', 'USGOLD' ]
symbol = symbols[0]
for symbol in symbols:
  #%% Create a directory
  ad = True
  ad_str = '_ad' if ad else ''
  directory_evals = f'N:/My Drive/Trading/Strategies/swing_vwap{ad_str}/{symbol}'
  directory_plots: str = f'N:/My Drive/Trading/Plots/swing_vwap{ad_str}/{symbol}_eval'
  os.makedirs(directory_plots, exist_ok=True)

  files = glob.glob(f'{directory_evals}/*.pkl')
  #%%
  print(f'Processing {symbol}...')
  results = []
  # Load a pickle file
  for i, file in enumerate(files):
    with open(file, 'rb') as f:  # Replace 'file.pkl' with your pickle file path
      data = pickle.load(f)
      data['date'] = data['VWAP'].index[0].date()
      data['VWAP_PCT'] = data['VWAP'].apply(lambda x: utils.pct.percentage_change(x, data['VWAP'].iat[0]))
      data['VWAP_PCT'].index = data['VWAP_PCT'].index.time
      day_change = data['VWAP_PCT'].iat[-1]
      prior_day_change = np.nan if i < 1 else utils.pct.percentage_change(results[i-1]['VWAP'].iat[-1], data['VWAP'].iat[-1])
      data['day_type'] = 'neutral' if abs(day_change) < 0.5 else 'up' if day_change > 0 else 'down'
      data['prior_day_change'] = prior_day_change
      data['day_type_prior'] = 'neutral' if abs(prior_day_change) < 0.5 else 'up' if prior_day_change > 0 else 'down'
      data['day_change'] = prior_day_change
      data['first_bars'] = data['firstBars']
      data['df_bracket_moves']['gap'] = 0 if np.isnan(prior_day_change) else prior_day_change
      data['df_bracket_moves_after_next']['gap'] = 0 if np.isnan(prior_day_change) else prior_day_change
      results.append(data)

  #%%
  def add_time_date(df, index_name = 'ts'):
    df.set_index(index_name, inplace=True)
    df['date'] = df.index.date
    df['time'] = df.index.time
    return df


#%%
  # Evaluate the time of the extrema
  extrema = [data['extrema'] for data in results]
  df_uptrends = add_time_date(pd.concat([data['uptrends'] for data in results]), 'start')
  df_uptrends['mid'] = (df_uptrends.o + df_uptrends.c)/2
  df_uptrends['type'] =  pd.cut(df_uptrends['bars'], bins=[-np.inf, 3, 5, np.inf], labels=['x<3', '3<x<5', 'x>5' ])

  df_downtrends = add_time_date(pd.concat([data['downtrends'] for data in results]), 'start')
  df_downtrends['type'] =  pd.cut(df_downtrends['bars'], bins=[-np.inf, 3, 5, np.inf], labels=['x<3', '3<x<5', 'x>5' ])
  df_downtrends['mid'] = (df_downtrends.o + df_downtrends.c)/2
  df_extrema = pd.concat(extrema)
  df_extrema = add_time_date(df_extrema, 'ts')

  df_extrema_lhd = df_extrema[~df_extrema.type.isin(['o', 'c'])].copy()

  df_extrema_lh = df_extrema[df_extrema.type.isin(['l', 'h'])].copy()
  df_extrema_lh['is_first_of_day'] = df_extrema_lh.groupby('date')['time'].transform('first') == df_extrema_lh['time']

  df_extrema_d_30 = df_extrema[df_extrema.type.isin(['dev30'])].copy()

#%%
  def process_bracket_moves(df_moves):
    df_moves.set_index('ts', inplace=True)
    df_moves['time'] = df_moves.index.time
    df_moves['sl_atr_pct_offset'] = df_moves.sl_pts_offset*100/df_moves.candle_atr_pts
    df_moves['trend_sentiment'] = (df_moves.candle_sentiment * df_moves.in_trend.astype(int).replace(0, -1))
    df_moves['bracket_sl_pts_offset'] = df_moves.sl_pts_offset + (df_moves.sl_value - df_moves.bracket_sl_value)*df_moves.trend_sentiment
    df_moves['bracket_sl_atr_pct_offset'] = df_moves.bracket_sl_pts_offset*100/df_moves.candle_atr_pts
    df_moves.loc[np.isinf(df_moves.bracket_sl_atr_pct_offset),'bracket_sl_atr_pct_offset'] = 10000.0
    df_moves.loc[np.isnan(df_moves.bracket_sl_atr_pct_offset),'bracket_sl_atr_pct_offset'] = 0.0
    df_moves['sl_atr_pct_offset'] = df_moves.sl_pts_offset*100/df_moves.candle_atr_pts
    df_moves.loc[np.isinf(df_moves.sl_atr_pct_offset),'sl_atr_pct_offset'] = 10000.0
    df_moves.loc[np.isnan(df_moves.sl_atr_pct_offset),'sl_atr_pct_offset'] = 0.0

    df_moves['in_trend'] = df_moves.in_trend.astype(bool)
    df_moves['bracket_entry_in_trend'] = df_moves.bracket_trigger == df_moves.trend_sentiment
    return df_moves


  df_candle_moves = process_bracket_moves(pd.concat([data['df_bracket_moves'] for data in results]))
  df_candle_moves_after_next = process_bracket_moves(pd.concat([data['df_bracket_moves_after_next'] for data in results]))

#%%
  def format_axes(ax):
    for row in ax:
      for a in row:
        labels = [label.get_text().replace('True', 'T').replace('False', 'F') for label in a.get_xticklabels()]
        a.set_xticklabels(labels, rotation=0)

        for container in a.containers:
          a.bar_label(container, fmt='%.2f', label_type='edge', padding=-10)


#%%
  time = df_candle_moves.time.unique()[3]
  stats =[('percentile_25', lambda x: np.nanpercentile(x, 25)), ('percentile_50', lambda x: np.nanpercentile(x, 50)), ('percentile_75', lambda x: np.nanpercentile(x, 75)) ]
  stats_stacked =[('percentile_25', lambda x: np.nanpercentile(x, 25)), ('percentile_50', lambda x: np.nanpercentile(x, 50) - np.nanpercentile(x, 25)), ('percentile_75', lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 50)) ]
  stats_labels=['25%', '50%', '75%']

  def plot_stats_row(df_candle, ax, type):
    # 1st evaluation: Probability of Candle to be in trend, overall points & SL
    df_candle.groupby(['in_trend', 'trend_sentiment']).agg({'pts_move': stats}).plot(kind='bar', ax = ax[0], colormap='Set3')
    ax[0].set_title(f'{type} Gains')
    ax[0].legend(labels=stats_labels)

    ax[1].set_title(f'{type} SL pts')
    df_candle.groupby(['in_trend', 'trend_sentiment']).agg({'sl_pts_offset': stats}).plot(kind='bar', ax = ax[1],colormap='Set3')
    ax[1].legend(labels=stats_labels)

    ax[2].set_title(f'{type} SL pct of atr')
    df_candle.groupby(['in_trend', 'trend_sentiment']).agg({'sl_atr_pct_offset': stats}).plot(kind='bar', ax = ax[2], colormap='Set3')
    ax[2].legend(labels=stats_labels)

    ax[3].set_title(f'{type} Candle Atr')
    df_candle.groupby(['in_trend', 'trend_sentiment']).agg({'candle_atr_pts': stats}).plot(kind='bar', ax = ax[3], colormap='Set3')
    ax[3].legend(labels=stats_labels)

  def plot_bracket_stats(df_candle, ax, type):
    # plot_stats_row(df_bracket_moves_in_trend, ax[0], 'All')
    df_candle[df_candle.bracket_entry_in_trend].groupby(['trend_sentiment']).agg({'pts_move': stats}).plot(kind='bar',  ax = ax[0], colormap='Set3')
    ax[0].set_title(f'{type} Gains per Trade in Trend')
    ax[0].legend(labels=stats_labels)

    df_candle[df_candle.bracket_entry_in_trend].groupby(['trend_sentiment']).agg({'bracket_sl_pts_offset': stats}).plot(kind='bar',  ax = ax[1], colormap='Set3')
    ax[1].set_title(f'{type} Necessary SL in Trend')
    ax[1].legend(labels=stats_labels)

    df_candle[df_candle.bracket_entry_in_trend].groupby(['trend_sentiment']).agg({'bracket_sl_atr_pct_offset': stats}).plot(kind='bar',  ax = ax[2], colormap='Set3')
    ax[2].set_title(f'{type} Necessary SL PCT of ATR in Trend')
    ax[2].legend(labels=stats_labels)

    if not df_candle[~df_candle.bracket_entry_in_trend].empty:
      df_candle[~df_candle.bracket_entry_in_trend].groupby(['trend_sentiment']).agg({'bracket_sl_pts_offset': stats}).plot(kind='bar',  ax = ax[3], colormap='Set3')
      ax[3].set_title(f'{type} Counter Trend in Profit')
      ax[3].legend(labels=stats_labels)
    ax[3].set_title(f'{type} NO Counter Trend in Profit')
  #%%
  directory_plots_candles = f"{directory_plots}/candles"
  directory_stats_candles = f"{directory_plots}/candle_stats"
  os.makedirs(directory_plots_candles, exist_ok=True)
  os.makedirs(directory_stats_candles, exist_ok=True)
  candle_move_stats = []
  for time in df_candle_moves.time.unique():
    #%%
    print(f"{datetime.datetime.now()}: Processing {time}")
    df_candle = df_candle_moves[df_candle_moves.index.time == time]
    df_candle_after_next = df_candle_moves_after_next[df_candle_moves_after_next.index.time == time]
    df_candle.to_csv(f'{directory_stats_candles}/{symbol}_{time.strftime("%H_%M")}_data.csv')
    # df_candle_after_next.to_csv(f'{directory_plots}/{symbol}_{time.strftime("%H_%M")}_data_after_next.csv')
    all_count = len(df_candle)
    all_count_after_next = len(df_candle_after_next)

    in_trend_moves_pts = df_candle[df_candle.bracket_entry_in_trend].pts_move.sum()
    in_trend_count_pct = len(df_candle[df_candle.bracket_entry_in_trend])*100/all_count
    not_in_trend_count_pct = len(df_candle[~df_candle.bracket_entry_in_trend])*100/all_count
    not_in_trend_candle_atr_pts = df_candle[~df_candle.bracket_entry_in_trend].candle_atr_pts.abs().sum()
    sl_in_trend = df_candle[df_candle.bracket_entry_in_trend].sl_pts_offset.describe()
    in_trend_moves_pts_stats = df_candle[df_candle.bracket_entry_in_trend].pts_move.describe()
    sl_in_trend_pct = df_candle[df_candle.bracket_entry_in_trend].bracket_sl_atr_pct_offset.describe()
    atr_not_in_trend = df_candle[df_candle.bracket_entry_in_trend].candle_atr_pts.describe()
    bct_in_profit = df_candle[~df_candle.bracket_entry_in_trend].bracket_sl_pts_offset.describe()

    candle_move_stats.append({ "time": time, "in_trend_moves_pts": in_trend_moves_pts, "in_trend_count_pct":in_trend_count_pct,
                               "not_in_trend_count_pct": not_in_trend_count_pct,
                               "not_in_trend_candle_atr_pts": not_in_trend_candle_atr_pts,
                               "in_trend_moves_pts_stats_25": in_trend_moves_pts_stats['25%'],
                               "in_trend_moves_pts_stats_50": in_trend_moves_pts_stats['mean'],
                               "in_trend_moves_pts_stats_75": in_trend_moves_pts_stats['75%'],
                               "sl_in_trend_25": sl_in_trend['25%'],
                               "sl_in_trend_50": sl_in_trend['mean'],
                               "sl_in_trend_75": sl_in_trend['75%'],
                               "sl_in_trend_pct_25": sl_in_trend_pct['25%'],
                               "sl_in_trend_pct_50": sl_in_trend_pct['mean'],
                               "sl_in_trend_pct_75": sl_in_trend_pct['75%'],
                               "bct_in_profit_25": bct_in_profit['25%'],
                               "bct_in_profit_50": bct_in_profit['mean'],
                               "bct_in_profit_75": bct_in_profit['75%'],
                               "atr_in_trend_no_hold_25": atr_not_in_trend['25%'],
                               "atr_in_trend_no_hold_50": atr_not_in_trend['mean'],
                               "atr_in_trend_no_hold_75": atr_not_in_trend['75%']})


  #%%
    fig, ax  = plt.subplots(2, 4, figsize=(19, 11), tight_layout=True)
    fig.suptitle(f'''
{symbol} {time} Stats
In Trend {df_candle.in_trend.sum()*100/all_count:.1f}%, Counter {(~df_candle.in_trend).sum()*100/all_count:.1f}%
In Next Trend {df_candle_after_next.in_trend.sum()*100/all_count_after_next:.1f}%, Counter {(~df_candle_after_next.in_trend).sum()*100/all_count_after_next:.1f}%
''')

    plot_stats_row(df_candle, ax[0], 'Bull/Bear Next')
    plot_stats_row(df_candle_after_next, ax[1], 'Bull/Bear After Next')

    format_axes(ax)
    # plt.show()
    plt.savefig(f'{directory_plots_candles}/{symbol}_{time.strftime("%H_%M")}_CandleStats.png', bbox_inches='tight')  # High-quality save
    plt.close()
    #%% Evaluate candle move where SL holds
    # 1st strategy: Bracket the Candle with SL on the opposite side
    fig, ax  = plt.subplots(2, 4, figsize=(19, 11), tight_layout=True)
    success_rate = df_candle.bracket_entry_in_trend.sum()*100/all_count
    success_rate_next_trend = df_candle_after_next.bracket_entry_in_trend.sum()*100/all_count
    fig.suptitle(f'''
{symbol} {time} Bracket Moves
Success {success_rate:.1f}%, Failure {100 - success_rate:.1f}% | Success Next Trend {success_rate:.1f}%, Failure {100 - success_rate:.1f}%
SL Below/Above gain {df_candle[df_candle.bracket_entry_in_trend].pts_move.sum():.0f}pts / loss {df_candle[~df_candle.bracket_entry_in_trend].candle_atr_pts.sum():.0f}pts | gain {df_candle_after_next[df_candle_after_next.bracket_entry_in_trend].pts_move.sum():.0f}pts / loss {df_candle_after_next[~df_candle_after_next.bracket_entry_in_trend].candle_atr_pts.sum():.0f}pts
''')
    plot_bracket_stats(df_candle, ax[0], 'Trend Move stats')
    plot_bracket_stats(df_candle_after_next, ax[1], 'Next Trend Move stats')

    format_axes(ax)
    # plt.show()

    plt.savefig(f'{directory_plots_candles}/{symbol}_{time.strftime("%H_%M")}_BracketMove.png', bbox_inches='tight')  # High-quality save
    plt.close()
 #%%
  def format_axes_2(ax, fct):
    for i,a in enumerate(ax):
      labels = [label.get_text().replace('True', 'T').replace('False', 'F') for label in a.get_xticklabels()]
      a.set_xticklabels(labels, rotation=90)
      a.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,  # Number of columns
        columnspacing=1.0,  # Space between columns
        frameon=True,  # Show frame
      )
      # Add styled grid
      a.grid(True, linestyle='--', color='gray', alpha=0.7, linewidth=0.5)
      a.spines[['top', 'right']].set_alpha(0.3)

      for container in a.containers:
          a.bar_label(container, fmt=fct[i], padding=3, rotation=90)
  #%%
  df_candle_move_stats = pd.DataFrame(candle_move_stats)
  df_candle_move_stats.set_index('time', inplace=True)
  df_candle_move_stats.sort_index(inplace=True)

  #%%
  fig, ax  = plt.subplots(3, 1, figsize=(38, 11), tight_layout=True)
  df_candle_move_stats.plot(kind='bar', y=['in_trend_moves_pts', 'not_in_trend_candle_atr_pts'], ax=ax[0])
  ax[0].set_title(f"{symbol} {time} In trend gains vs counter trend losses")
  df_candle_move_stats.plot(kind='bar', y=['in_trend_count_pct'], ax=ax[1])
  ax[1].set_title(f"{symbol} {time} Bracket entry in trend success pct")
  df_candle_move_stats.plot(kind='bar', y=['in_trend_moves_pts_stats_25', 'in_trend_moves_pts_stats_50', 'in_trend_moves_pts_stats_75'], ax=ax[2])
  ax[2].set_title(f"{symbol} {time} Stats of trend move")
  format_axes_2(ax, [lambda x: humanize.naturalsize(x).replace('B', ''), '%.1f', '%.0f'])

#  plt.show()
  plt.savefig(f'{directory_plots}/{symbol}_MovesInTrendPts.png', bbox_inches='tight')  # High-quality save
  plt.close()
  #%%

  fig, ax  = plt.subplots(2, 1, figsize=(38, 11), tight_layout=True)
  df_candle_move_stats.plot(kind='bar', y=['bct_in_profit_25','bct_in_profit_50','bct_in_profit_75'], ax=ax[0])
  ax[0].set_title(f"{symbol} {time} Stats of counter trend moves in profit")
  df_candle_move_stats.plot(kind='bar', y=['atr_in_trend_no_hold_25', 'atr_in_trend_no_hold_50', 'atr_in_trend_no_hold_75'], ax=ax[1])
  ax[1].set_title(f"{symbol} {time} Stats of counter trend losses when SL is below bar")

  format_axes_2(ax, ['%.0f', '%.0f'])
  # plt.show()
  plt.savefig(f'{directory_plots}/{symbol}_MovesCounterTrend.png', bbox_inches='tight')  # High-quality save
  plt.close()
  #%%

  fig, ax  = plt.subplots(2, 1, figsize=(38, 11), tight_layout=True)
  df_candle_move_stats.plot(kind='bar', y=['sl_in_trend_25', 'sl_in_trend_50', 'sl_in_trend_75'], ax=ax[0])
  ax[0].set_title(f"{symbol} {time} Stats of necessary SL for in trend moves")
  df_candle_move_stats.plot(kind='bar', y=['sl_in_trend_pct_25', 'sl_in_trend_pct_50', 'sl_in_trend_pct_75'], ax=ax[1])
  ax[1].set_title(f"{symbol} {time} Stats in pct of necessary SL for in trend moves")

  format_axes_2(ax, ['%.0f', '%.0f'])
#  plt.show()
  plt.savefig(f'{directory_plots}/{symbol}_MovesInTrendSL.png', bbox_inches='tight')  # High-quality save
  plt.close()

  #%% Calculate channels
  def channels(df, is_larger = True, pb_offset = 0.5):
    channels = []
    channel = None
    for i in range(0, len(df)):
      if channel is not None and (df.iloc[i].name.date() == channel['end'].date() and
           ((is_larger and df.iloc[i].o > channel['SL']) or
            (not is_larger and df.iloc[i].o < channel['SL']))):
        is_range_extended = is_larger and df.iloc[i].c > channel['c'] or not is_larger and df.iloc[i].c < channel['c']
        channel['end'] = df.iloc[i].end
        channel['c'] = df.iloc[i].c if is_range_extended else channel['c']
        channel['bars'] += df.iloc[i].bars
        channel['ema20_c'] = df.iloc[i].ema20_c
        channel['legs'] += 1
        channel['hlegs'] += 1 if is_range_extended else 0
        channel['SL'] = channel['o'] + (channel['c'] - channel['o'])*pb_offset
      else:
        if channel is not None:
          channels.append(channel)
        channel = df.iloc[i].to_dict()
        channel['legs'] = 1
        channel['hlegs'] = 1
        channel['SL'] = channel['o'] + (channel['c'] - channel['o'])*pb_offset

    if channel is not None:
      channels.append(channel)
    df_channels = pd.DataFrame(channels)
    # df_channels.set_index('start', inplace=True)
    return df_channels
#%%
  dfs_bull_channels = []
  dfs_bear_channels = []
  for i in range(11):
    dfs_bull_channels.append({str(i): channels(df_uptrends, is_larger=True, pb_offset=i/10)})
    dfs_bear_channels.append({str(i): channels(df_downtrends, is_larger=False, pb_offset=i/10)})

  #%%
  df = df_extrema_lhd
  def plot_type_percentages(df, event = ''):
    # Calculate percentages for each type independently
    type_counts = df.groupby(['time', 'type'], observed=True).size().unstack(fill_value=0)
    type_percentages = type_counts.div(type_counts.sum(), axis=1) * 100

    # Calculate percentages for each type independently
    all_counts = df.groupby(['time']).size()
    all_percentages = all_counts.div(all_counts.sum()) * 100
    type_percentages['all'] = all_percentages

    # Calculate cumulative percentages for each type independently
    cumulative_counts = type_counts.cumsum()
    cumulative_percentages = cumulative_counts.div(type_counts.sum(), axis=1) * 100

    all_cumulative_counts = all_counts.cumsum()
    all_cumulative_percentages = all_cumulative_counts.div(all_counts.sum()) * 100
    cumulative_percentages['all'] = all_cumulative_percentages

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(38, 11))

    # Plot 1: Individual type percentages
    type_percentages.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title(f'{symbol} Distribution of {len(df)} {event} Events by Type\n(Each type adds to 100%)')
    ax1.set_xlabel('')
    ax1.set_ylabel('Percentage of Type Events')
    ax1.legend(title='Type')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add percentage labels on the bars
    # for container in ax1.containers:
    #   ax1.bar_label(container, fmt='%.1f%%', padding=3)

    # Plot 2: Cumulative percentages
    cumulative_percentages.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title(f'{symbol} Cumulative Distribution of {len(df)} {event} Events by Type\n(Each type reaches 100%)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Percentage of Type Events')
    ax2.legend(title='Type')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    for a in [ax1, ax2]:
      labels = [label.get_text().replace('True', 'T').replace('False', 'F') for label in a.get_xticklabels()]
      a.set_xticklabels(labels, rotation=90)

      # Add styled grid
      a.grid(True, linestyle='--', color='gray', alpha=0.7, linewidth=0.5)
      for container in a.containers:
        a.bar_label(container, fmt='%.2f', padding=3, rotation=90)

    # Add percentage labels on the bars
    # for container in ax2.containers:
    #   ax2.bar_label(container, fmt='%.1f%%', padding=3)

    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', rotation=90)

    # Adjust layout to prevent overlap
    plt.tight_layout()

  #%%
  print('Plotting...')
  plot_type_percentages(df_extrema_lh[df_extrema_lh.is_first_of_day], 'First of Day')
  plt.savefig(f'{directory_plots}/{symbol}_FirstOfDay.png', bbox_inches='tight')  # High-quality save
  plt.close()

  plot_type_percentages(df_extrema_lh[~df_extrema_lh.is_first_of_day], 'Second of Day')
  plt.savefig(f'{directory_plots}/{symbol}_SecondOfDay.png', bbox_inches='tight')  # High-quality save
  plt.close()

  plot_type_percentages(df_extrema_d_30, 'Deviation 30%')
  plt.savefig(f'{directory_plots}/{symbol}_Deviations_30.png', bbox_inches='tight')  # High-quality save
  plt.close()

  plot_type_percentages(df_downtrends, 'Downtrends')
  plt.savefig(f'{directory_plots}/{symbol}_Downtrends.png', bbox_inches='tight')  # High-quality save
  plt.close()

  plot_type_percentages(df_downtrends[~df_downtrends.trend_support], 'Downtrends With Trend')
  plt.savefig(f'{directory_plots}/{symbol}_DowntrendsWithTrend.png', bbox_inches='tight')  # High-quality save
  plt.close()

  plot_type_percentages(df_downtrends[~df_downtrends.trend_support], 'Downtrends Against Trend')
  plt.savefig(f'{directory_plots}/{symbol}_DowntrendsAgainstTrend.png', bbox_inches='tight')  # High-quality save
  plt.close()

  plot_type_percentages(df_uptrends, 'Uptrends')
  plt.savefig(f'{directory_plots}/{symbol}_Uptrends.png', bbox_inches='tight')  # High-quality save
  plt.close()

  plot_type_percentages(df_uptrends[~df_uptrends.trend_support], 'Uptrends With Trend')
  plt.savefig(f'{directory_plots}/{symbol}_UptrendsWithTrend.png', bbox_inches='tight')  # High-quality save
  plt.close()

  plot_type_percentages(df_uptrends[~df_uptrends.trend_support], 'Uptrends Against Trend')
  plt.savefig(f'{directory_plots}/{symbol}_UptrendsAgainstTrend.png', bbox_inches='tight')  # High-quality save
  plt.close()


#%%
#  # plt.show()

#%%
  all_vwap_pct = {}
  for day_type in ['up', 'neutral', 'down']:
    # all_vwap_pct[day_type] = pd.concat([result['VWAP_PCT'] for result in results if result['day_type'] == day_type])
    all_vwap_pct[day_type] = pd.concat([result['VWAP_PCT'] for result in results if result['day_type_prior'] == day_type])


  #%%
  # Create the boxplot
  fig, axes = plt.subplots(3, 1, figsize=(20, 14), tight_layout=True)  # 3 rows, 1 column

  for i, day_type in enumerate(['up', 'neutral', 'down']):
    grouped = all_vwap_pct[day_type].groupby(all_vwap_pct[day_type].index)

    # Prepare the data for the boxplot
    boxplot_data = [group for _, group in grouped]

    axes[i].boxplot(boxplot_data, positions=range(len(grouped)), widths=0.6, patch_artist=True, showfliers=False, notch=True)
    # axes[i].violinplot(boxplot_data, positions=range(len(grouped)), widths=0.6, showmeans=True, showmedians=True, showextrema=False)
    # Format the x-axis with time labels
    axes[i].set_xticks(range(len(grouped)), [time.strftime('%H:%M') for time in grouped.groups.keys()], rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    # axes[i].set_ylim(-2, 2)
  fig.suptitle("Boxplot of VWAP percentage changes")
#  # plt.show()
  plt.savefig(f'{directory_plots}/{symbol}_VWAP_PCT.png', bbox_inches='tight')  # High-quality save
  plt.close()

  print('Done.')




#%%
# Plot all vwap curves for up, neutral & down days into one chart
year = 2020
fig, axes = plt.subplots(3, 1, figsize=(20, 14), tight_layout=True)  # 3 rows, 1 column
alpha = 0.2
colors = {2020: 'blue', 2021: 'red', 2022: 'green', 2023: 'orange', 2024: 'purple', 2025: 'brown'}

for result in results:
  year = result['date'].year
  day_change_pct = result['VWAP_PCT'].iat[-1]
  if -0.5 < day_change_pct < 0.5:
    result['VWAP_PCT'].plot.line(alpha=alpha, color=colors[year], ax=axes[1])
  elif day_change_pct < -0.5:
    result['VWAP_PCT'].plot.line(alpha=alpha, color=colors[year], ax=axes[0])
  else:
    result['VWAP_PCT'].plot.line(alpha=alpha, color=colors[year], ax=axes[2])

for ax in axes:
  ax.axhline(y=0, color='grey', linestyle='--', linewidth=1, label='Zero Line')
  ax.set_ylim(-2, 2)
#plt.show()



### Plot of counts over time, not so useful
#%%
def plot_type_counts(df, suptitle = ''):
  count_df = df.groupby(['time', 'type']).size().unstack(fill_value=0)

  # Create the bar chart
  ax = count_df.plot(kind='bar', figsize=(24, 14))

  # Customize the chart
  plt.title('Frequency of Types Over Time')
  plt.suptitle(suptitle)
  plt.xlabel('Time')
  plt.ylabel('Count')
  plt.legend(title='Type')
  plt.xticks(rotation=45)
  plt.tight_layout()

#  plt.show()

#%%
# Count occurrences of each type per time
df_extrema_lhd = df_extrema[~df_extrema.type.isin(['o', 'c'])].copy()

plot_type_counts(df_extrema_lh, 'All Lows/Highs')

#%%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 14), tight_layout=True)
df_extrema_lh[df_extrema_lh.is_first_of_day].groupby(['time', 'type']).size().unstack(fill_value=0).plot(kind='bar', ax=ax1)
df_extrema_lh[~df_extrema_lh.is_first_of_day].groupby(['time', 'type']).size().unstack(fill_value=0).plot(kind='bar', ax=ax2)
df_extrema_d.groupby(['time', 'type']).size().unstack(fill_value=0).plot(kind='bar', ax=ax3)

for ax in [ax1, ax2, ax3]:
  ax.set_xlabel('Time')
  ax.set_ylabel('Count')
  ax.legend(title='Type')
  ax.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.title('Frequency of Types Over Time')
ax1.set_title('First of Day')
ax2.set_title('Second of Day')
ax3.set_title('Deviations')
#plt.show()
