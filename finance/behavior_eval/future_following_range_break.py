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
directory = f'N:/My Drive/Trading/Strategies/future_following_range_break'
os.makedirs(directory, exist_ok=True)
# symbol = 'IBDE40'
# symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
#            *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]
symbols = ['IBDE40', 'IBGB100', 'IBES35', 'IBJP225', 'IBUS30', 'IBUS500', 'IBUST100']

# symbol = symbols[0]
for symbol in symbols[1:]:
  # Create a directory
  symbol_directory = f'{directory}/{symbol}'
  os.makedirs(symbol_directory, exist_ok=True)
  symbol_def = utils.influx.SYMBOLS[symbol]
  tz = symbol_def['EX']['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = tz.localize(dateutil.parser.parse('2023-01-02T00:00:00'))
  last_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
  day_start = first_day
  prior_close = None
  df_raw = None
  timeranges = ['2m', '10m', '5m']
  # timerange = '10m'
  follow = {'start':None, 'end':None, 'low': None, 'high': None, 'stopout': None, 'candles':None, 'strategy': None}
  S_01_pct = '01_pct'
  S_02_pct = '02_pct'
  S_cbc = 'cbc'
  S_cbc_10_pct = 'cbc_10_pct'
  S_cbc_20_pct = 'cbc_20_pct'
  S_cbc_10_pct_up = 'cbc_10_pct_up'
  S_cbc_20_pct_up = 'cbc_20_pct_up'

  strategies = [S_01_pct, S_02_pct, S_cbc, S_cbc_10_pct, S_cbc_20_pct, S_cbc_10_pct_up, S_cbc_20_pct_up]
  # strategies = [S_01_pct, S_cbc, S_cbc_10_pct, S_cbc_10_pct_up]
  ## %%
  while day_start < last_day:
    for timerange in timeranges:
    # get the following data for daily assignment
      df_day = utils.influx.get_candles_range_aggregate(day_start + symbol_def['EX']['Open'], day_start + symbol_def['EX']['Close'], symbol, timerange)
      day_end = day_start + datetime.timedelta(days=1)
      if df_day is None:
        print(f'no data for {day_start.isoformat()}')
        day_start = day_end
        continue

      ##%%
      follow = []
      open_long = {}
      open_short ={}
      ##%%
      for i in range(1, len(df_day)):
        last_candle = df_day.iloc[i - 1]
        candle = df_day.iloc[i]
        last_atr = last_candle.h - last_candle.l
        offsets = {S_01_pct: last_candle.o * 0.001, S_02_pct: last_candle.o * 0.002, S_cbc: 0, S_cbc_10_pct: last_atr * 0.1,
                   S_cbc_20_pct: last_atr * 0.2, S_cbc_10_pct_up: -last_atr * 0.1, S_cbc_20_pct_up: -last_atr * 0.2,}

        for strategy in strategies:
          # Long entry or follow through
          is_stop_long = False
          if strategy not in open_long and last_candle.h < candle.h:
            low = last_candle.l if last_candle.l < candle.l else candle.l
            open_long[strategy] = {'start': last_candle.name, 'end': None, 'last_atr': last_atr, 'low': low,
                                   'entry': last_candle.h, 'high': candle.h, 'stopout': 1, 'candles': 1, 'strategy': strategy,
                                   'type': 'long', 'high_1':candle.h}
          elif strategy in open_long:
            high = candle.h if candle.h > last_candle.h else last_candle.h
            if last_candle.l - offsets[strategy] < candle.l:
              open_long[strategy]['candles'] += 1
              open_long[strategy]['high'] = high
              ith_candle = open_long[strategy]["candles"]
              if ith_candle < 5:
                open_long[strategy][f'high_{ith_candle}'] = candle.h
            else:
              is_stop_long = True
              open_long[strategy]['stopout'] = last_candle.l - offsets[strategy]

            if ~is_stop_long and i == len(df_day) - 1:
              is_stop_long = True
              open_long[strategy]['stopout'] = last_candle.h

          if is_stop_long:
            open_long[strategy]['end'] = candle.name
            open_long[strategy]['loss'] = open_long[strategy]['stopout'] < open_long[strategy]['entry']
            follow.append(open_long[strategy])
            del open_long[strategy]

          # Long entry or follow through
          is_stop_short = False
          if strategy not in open_short and last_candle.l > candle.l:
            high = last_candle.h if last_candle.h > candle.h else candle.h
            open_short[strategy] = {'start': last_candle.name, 'end': None, 'last_atr': last_atr, 'low': candle.l,
                                    'entry': last_candle.l, 'high': high, 'stopout': 1, 'candles': 1, 'strategy': strategy,
                                    'type': 'short', 'low_1':candle.l}
          elif strategy in open_short:
            low = candle.l if candle.l < last_candle.l else last_candle.l
            if last_candle.h + offsets[strategy] > candle.h:
              open_short[strategy]['candles'] += 1
              open_short[strategy]['low'] = low
              ith_candle = open_short[strategy]["candles"]
              if ith_candle < 5:
                open_short[strategy][f'low_{ith_candle}'] = candle.l
            else:
              is_stop_short = True
              open_short[strategy]['stopout'] = last_candle.h + offsets[strategy]

            if ~is_stop_short and i == len(df_day) - 1:
              is_stop_short = True
              open_short[strategy]['stopout'] = last_candle.h

          if is_stop_short:
            open_short[strategy]['end'] = candle.name
            open_short[strategy]['loss'] = open_short[strategy]['stopout'] > open_short[strategy]['entry']
            follow.append(open_short[strategy])
            del open_short[strategy]

      df_follow = pd.DataFrame(follow)
      df_follow.to_pickle(f'{symbol_directory}/{symbol}_{timerange}_{day_start.strftime("%Y-%m-%d")}_follow.pkl')
      print(f'Evaluated: {symbol} {timerange} {day_start.strftime("%Y-%m-%d")}')
    day_start = day_end

# %%
mpf.plot(df_day, style='yahoo', figsize=(20, 12), tight_layout=True, type='candle',
         columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M')
date_str = day_start.strftime('%Y-%m-%d')
prior_close_str = f'Prior Close: {prior_close:.2f}' if prior_close is not None else 'N/A'
plt.gcf().suptitle(f'{symbol} {date_str} {timerange} {prior_close_str}')

#%%
longs = df_follow[(df_follow['type']=='long') & (df_follow['strategy']==S_cbc)]
longs_df = pd.DataFrame(index=df_day.index)
longs_df['entry'] = float('nan')
longs_df.loc[longs['start'].tolist(), 'entry'] = longs['entry'].tolist()# Fill
longs_df['exit'] = float('nan')
longs_df.loc[longs['end'].tolist(), 'exit'] = longs['stopout'].tolist()# Fill

shorts = df_follow[(df_follow['type']=='short') & (df_follow['strategy']==S_cbc)]
shorts_df = pd.DataFrame(index=df_day.index)
shorts_df['entry'] = float('nan')
shorts_df.loc[shorts['start'].tolist(), 'entry'] = shorts['entry'].tolist()# Fill
shorts_df['exit'] = float('nan')
shorts_df.loc[shorts['end'].tolist(), 'exit'] = shorts['stopout'].tolist()# Fill

add_plot = [
  mpf.make_addplot(longs_df['entry'], color="blue", marker='^', type="scatter", width=1.5),
  mpf.make_addplot(longs_df['exit'], color="orange", marker="v", type="scatter", width=1.5),
  mpf.make_addplot(shorts_df['entry'], color="cyan", marker='v', type="scatter", width=1.5),
  mpf.make_addplot(shorts_df['exit'], color="brown", marker='^', type="scatter", width=1.5),
]

mpf.plot(df_day, style='yahoo', addplot=add_plot, figsize=(20, 12), tight_layout=True, type='candle',
         columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M')
date_str = day_start.strftime('%Y-%m-%d')
prior_close_str = f'Prior Close: {prior_close:.2f}' if prior_close is not None else 'N/A'
plt.gcf().suptitle(f'{symbol} {date_str} {timerange} {prior_close_str}')
plt.show()
# # plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
# # plt.savefig(f'finance/_data/dax_plots_mpl/IBDE40_{date_str}.png', dpi=300, bbox_inches='tight')  # High-quality save
# plt.show()
# # plt.close()
