#%%
from datetime import datetime, timedelta


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
from scipy.stats import linregress

import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
symbols = ['IBDE40', 'IBES35', 'IBFR40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225']
# symbol = symbols[0]
for symbol in symbols:
  # Create a directory
  directory = f'N:/My Drive/Projects/Trading/Research/Plots/swing/{symbol}_5m_9ema_ama_slope'
  os.makedirs(directory, exist_ok=True)

  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]

  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = tz.localize(dateutil.parser.parse('2022-01-03T00:00:00'))
  # first_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
  now = datetime.now(tz)
  last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  ##%%
  offset = timedelta(hours=0)
  while day_start < last_day:
    #%%
    day_end = day_start + exchange['Close'] + timedelta(hours=1) + offset
    # get the following data for daily assignment
    day_candles = utils.influx.get_candles_range_raw(day_start+exchange['Open']-timedelta(hours=1, minutes=0)+offset, day_end, symbol)
    if day_candles is None or len(day_candles) < 100:
      day_start = day_start + timedelta(days=1)
      print(f'{symbol} no data for {day_start.isoformat()}')
      # raise Exception('no data')
      continue
    prior_day_candle = utils.influx.get_candles_range_aggregate(prior_day + exchange['Open'] + offset, prior_day + exchange['Close'] + offset, symbol)
    overnight_candle= utils.influx.get_candles_range_aggregate(day_start + offset, day_start + exchange['Open'] - timedelta(hours=1) + offset, symbol)

    prior_day = day_start
    day_start = day_start + timedelta(days=1)
  ##%%

    df_1m = day_candles
    df_5m = df_1m.resample('5min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    df_30m = df_1m.resample('30min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))

    df_5m['9EMA'] = df_5m['c'].ewm(span=9, adjust=False).mean()
    # Calculate the Adaptive Moving Average (AMA / KAMA)
    df_5m['AMA'] = utils.indicators.adaptive_moving_average(df_5m['c'], period=10, fast=2, slow=30)

    # Define a rolling window size (e.g., 10 days)
    window_size = 3

    # Function to calculate the slope for a window
    def calculate_slope(x):
      indices = np.arange(len(x))  # Create a simple index for regression (0, 1, 2, ...)
      slope, intercept, r_value, p_value, std_err = linregress(indices, x)
      return slope

    # Apply a rolling window to calculate the slope
    df_5m["SL_9EMA"] = df_5m["9EMA"].rolling(window=window_size).apply(calculate_slope, raw=True)
    df_5m["SL_AMA"] = df_5m["AMA"].rolling(window=window_size).apply(calculate_slope, raw=True)
  #%%
    try:
      # %%
      fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)

      date_str = day_start.strftime('%Y-%m-%d')
      ax1 = fig.add_subplot(2,1,1)
      ax2 = fig.add_subplot(2,1,2)

      overnight_h = overnight_candle.h.iat[0] if overnight_candle is not None else np.nan
      overnight_l = overnight_candle.l.iat[0] if overnight_candle is not None else np.nan

      indicator_hlines = [prior_day_candle.c.iat[0], prior_day_candle.h.iat[0], prior_day_candle.l.iat[0], overnight_h, overnight_l]
      fig.suptitle(f'{symbol} {date_str} 1m/5m/30m PriorDay: H {prior_day_candle.h.iat[0]:.2f}  C {prior_day_candle.c.iat[0]:.2f} L {prior_day_candle.l.iat[0]:.2f} On: H {overnight_h:.2f} L {overnight_l:.2f}')

      hlines=dict(hlines=indicator_hlines, colors=['#bf42f5'], linewidths=[0.5, 1, 1, 0.5, 0.5], linestyle=['--', *['-']*(len(indicator_hlines)-1)])

      ema_plot = mpf.make_addplot(df_5m['9EMA'], ax=ax1, width=1, color="turquoise")
      ama_plot = mpf.make_addplot(df_5m['AMA'], ax=ax1, width=1, color='gold')

      mpf.plot(df_5m, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=[ama_plot, ema_plot])

      ema_slope_plot = mpf.make_addplot(df_5m['SL_9EMA'], ax=ax2, width=1, color="turquoise")
      ama_slope_plot = mpf.make_addplot(df_5m['SL_AMA'], ax=ax2, width=1, color='gold')
      slope_df = df_5m.copy()

      mpf.plot(slope_df, type='line', ax=ax2,columns=['SL_AMA']*5,  xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=[ema_slope_plot, ama_slope_plot])

      # plt.show()
      ## %%
      plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
      plt.close()
      print(f'{symbol} finished {date_str}')
      #%%
    except Exception as e:
      print(f'{symbol} error: {e}')
      continue

