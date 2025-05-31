#%%
import pickle
from datetime import datetime, timedelta, time

import dateutil
import numpy as np
import scipy

import pandas as pd
import os

import influxdb as idb

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
from matplotlib import gridspec

import finance.utils as utils
from finance.behavior_eval.futures_vwap_extrema import day_data

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
symbols = ['ESTX50', 'SPX', 'INDU', 'NDX', 'N225']
symbol = symbols[0]
for symbol in symbols:
  #%% Create a directory
  directory_evals = f'N:/My Drive/Trading/Strategies/close_to_min/{symbol}'
  directory_plots = f'N:/My Drive/Trading/Plots/close_to_min/{symbol}'
  os.makedirs(directory_evals, exist_ok=True)
  os.makedirs(directory_plots, exist_ok=True)

  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]

  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = tz.localize(dateutil.parser.parse('2020-01-02T00:00:00'))
  # first_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
  now = datetime.now(tz)
  last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  ##%%
  day_data = utils.trading_day_data.TradingDayData(symbol, timedelta(days=2), timedelta(days=365))
  #%%
  while day_start < last_day:
#%%
    # find crossed candles for different max times
    day_data.update(day_start, prior_day)

    if not day_data.has_sufficient_data():
      print(f'Skipping day {day_start.date()} because of insufficient data')
      day_start = day_start + timedelta(days=1)
      continue
      # raise(Exception('Insufficient data'))

    all_dists = []
    min_time = day_start + exchange['Open']
    max_time = day_start + exchange['Open'] + timedelta(minutes=30)
    close_time = day_start + exchange['Close']
    if close_time > day_data.df_5m.index.max():
      close_time = day_data.df_5m.index.max()

    while max_time < close_time - timedelta(hours=3):
      filter_max_time = (day_data.df_5m.index.time >= min_time.time()) & (day_data.df_5m.index.time <= max_time.time())
      if not any(filter_max_time):
        min_time = max_time
        max_time = min_time + timedelta(minutes=30)
        continue
      df_5m_filtered = day_data.df_5m[filter_max_time]
      closest_indices = df_5m_filtered[(df_5m_filtered.l < day_data.cdc) & (day_data.cdc < df_5m_filtered.h)].index.to_list()
      if len(closest_indices) > 0:
        closest_index = min(closest_indices)
        dist = 0
        value = day_data.cdc
      else:
        ##%%
        closest_l = (df_5m_filtered.l - day_data.cdc).abs().min()
        closest_h = (df_5m_filtered.h - day_data.cdc).abs().min()
        if closest_l < closest_h:
           closest_index = (df_5m_filtered.l - day_data.cdc).abs().idxmin()
           value = day_data.df_5m.loc[closest_index].l
        else:
          closest_index = (df_5m_filtered.h - day_data.cdc).abs().idxmin()
          value = day_data.df_5m.loc[closest_index].h

        dist = min(closest_l, closest_h)/day_data.cdc
      all_dists.append({'ts': closest_index, 'date': closest_index.date(), 'weekday': closest_index.weekday(), 'value': value, 'cdc': day_data.cdc, 'time': closest_index.time(), 'dist': dist, 'minT': min_time.time(), 'maxT': max_time.time()})
      min_time = max_time
      max_time = min_time + timedelta(minutes=30)

##%%
    alines = []
    for dst in all_dists:
      alines.append([(dst['ts'], dst['value']), (day_data.df_5m.iloc[-2].name , dst['value'])])

    alines=dict(alines=alines,
              colors=['mediumblue'],
              alpha=0.5, linewidths=[0.25],
              linestyle=['-'])

    utils.plots.daily_change_plot(day_data, alines, '')
    # plt.show()

  #%%
    date_str = day_start.strftime('%Y-%m-%d')
    # metadata = {"dists": all_dists, "day": day_start}
    # with open(f'{directory_evals}/{symbol}_{date_str}.pkl', "wb") as f:
    #   pickle.dump(metadata, f)
    df_all_dists = pd.DataFrame(all_dists)
    df_all_dists.to_pickle(f'{directory_evals}/{symbol}_{date_str}.pkl')
    plt.savefig(f'{directory_plots}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
    plt.close()
    print(f'{symbol} finished {date_str}')
    #%%
    prior_day = day_start
    day_start = day_start + timedelta(days=1)

#%%

