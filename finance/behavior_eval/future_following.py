# %%
import datetime

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

import finance.utils as utils
pd.options.plotting.backend = "matplotlib"


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
directory = f'N:/My Drive/Projects/Trading/Research/Strategies/future_following'
os.makedirs(directory, exist_ok=True)
# symbol = 'IBDE40'
# symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
#            *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]
symbols = ['IBDE40', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100']
symbol = symbols[0]
# for symbol in symbols:
# %%
# Create a directory
symbol_directory = f'{directory}/{symbol}'
os.makedirs(symbol_directory, exist_ok=True)
symbol_def = utils.influx.SYMBOLS[symbol]
tz = symbol_def['EX']['TZ']

dfs_ref_range = []
dfs_closing = []
first_day = tz.localize(dateutil.parser.parse('2020-01-02T00:00:00'))
last_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
day_start = first_day
prior_close = None
df_raw = None
timerange = '10m'
follow = {'start':None, 'end':None, 'low': None, 'high': None, 'stopout': None, 'candles':None, 'strategy': None}
S_01_pct = '01_pct'
S_02_pct = '02_pct'
S_cbc = 'cbc'
S_cbc_10_pct = 'cbc_10_pct'
S_cbc_20_pct = 'cbc_20_pct'

strategies = [S_01_pct, S_02_pct, S_cbc, S_cbc_10_pct, S_cbc_20_pct]
# %%
while day_start < last_day:
# get the following data for daily assignment
  df_10m = utils.influx.get_candles_range_aggregate(day_start + symbol_def['EX']['Open'], day_start+symbol_def['EX']['Close'], symbol, timerange)
  day_end = day_start + datetime.timedelta(days=1)
  if df_10m is None:
    print(f'no data for {day_start.isoformat()}')
    day_start = day_end
    continue

  #%%
  follow = []
  open_long = {}
  open_short ={}
  ##%%
  for i in range(1, len(df_10m)):
    last_candle = df_10m.iloc[i-1]
    candle = df_10m.iloc[i]

    last_atr = last_candle.h - last_candle.l
    offsets = {S_01_pct: last_candle.o * 0.001, S_02_pct: last_candle.o * 0.002, S_cbc: 0, S_cbc_10_pct: last_atr * 0.1, S_cbc_20_pct: last_atr * 0.2}

    for strategy in strategies:
      # Long entry or follow through
      is_stop_long = False
      if last_candle.l - offsets[strategy] < candle.l:
        high = candle.h if candle.h > last_candle.h else last_candle.h
        if strategy not in open_long:
          open_long[strategy] =  {'start':last_candle.name, 'end':None, 'low': last_candle.l, 'high': high, 'stopout': 0, 'candles':2, 'strategy': strategy, 'type': 'long'}
        else:
          open_long[strategy]['candles'] += 1
          open_long[strategy]['high'] = high

        if i == len(df_10m)-1:
          is_stop_long = True
          open_long[strategy]['stopout'] = last_candle.h
      elif strategy in open_long:
        is_stop_long = True
        open_long[strategy]['stopout'] = last_candle.l - offsets[strategy]

      if is_stop_long:
        open_long[strategy]['end'] = candle.name
        follow.append(open_long[strategy])
        del open_long[strategy]

      is_stop_short = False
      if last_candle.h + offsets[strategy] > candle.h:
        low = candle.l if candle.l < last_candle.l else last_candle.l
        if strategy not in open_short:
          open_short[strategy] =  {'start':last_candle.name, 'end':None, 'low': low, 'high': last_candle.h, 'stopout': 0, 'candles':2, 'strategy': strategy, 'type': 'short'}
        else:
          open_short[strategy]['candles'] += 1
          open_short[strategy]['low'] = low
        if i == len(df_10m)-1:
          is_stop_short = True
          open_short[strategy]['stopout'] = last_candle.l
      elif strategy in open_short:
        open_short[strategy]['stopout'] = last_candle.h + offsets[strategy]
        is_stop_short = True

      if is_stop_short:
        open_short[strategy]['end'] = candle.name
        follow.append(open_short[strategy])
        del open_short[strategy]


  df_follow = pd.DataFrame(follow)
  df_follow.to_pickle(f'{symbol_directory}/{symbol}_{day_start.strftime("%Y-%m-%d")}_follow.pkl')
  day_start = day_end

# %%
# mpf.plot(df_10m, style='yahoo', figsize=(20, 12), tight_layout=True, type='candle',
#          columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M')
# date_str = day_start.strftime('%Y-%m-%d')
# prior_close_str = f'Prior Close: {prior_close:.2f}' if prior_close is not None else 'N/A'
# plt.gcf().suptitle(f'{symbol} {date_str} {timerange} {prior_close_str}')
#
# # plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
# # plt.savefig(f'finance/_data/dax_plots_mpl/IBDE40_{date_str}.png', dpi=300, bbox_inches='tight')  # High-quality save
# plt.show()
# # plt.close()
