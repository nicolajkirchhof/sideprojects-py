#%%
import re
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

import scipy.signal as signal
mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
symbols = ['IBDE40', 'IBES35', 'IBFR40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225']
symbol = symbols[0]

for symbol in symbols:
  #%%
  # Create a directory
  directory = f'N:/My Drive/Trading/Plots/swing/{symbol}_5m_9ema_ama_slope'
  os.makedirs(directory, exist_ok=True)

  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]

  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = dateutil.parser.parse('2022-01-03T00:00:00').replace(tzinfo=tz)

  # first_day_from_file = utils.plots.last_date_from_files(directory)
  # if first_day_from_file is not None:
  #   first_day = first_day_from_file.replace(tzinfo=tz)

  now = datetime.now(tz)
  last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  offset = timedelta(hours=0)
  #%%
  while day_start < last_day:
    #%%
    day_end = day_start + exchange['Close'] + timedelta(hours=1) + offset
    # get the following data for daily assignment
    day_candles = utils.influx.get_candles_range_raw(day_start+exchange['Open']-timedelta(hours=1, minutes=0)+offset, day_end, symbol)
    if day_candles is None or len(day_candles) < 100:
      day_start = day_start + timedelta(days=1)
      print(f'{symbol} no data for {day_start.isoformat()}')
      raise Exception('no data')
      # continue
    prior_day_candle = utils.influx.get_candles_range_aggregate(prior_day + exchange['Open'] + offset, prior_day + exchange['Close'] + offset, symbol)
    overnight_candle= utils.influx.get_candles_range_aggregate(day_start + offset, day_start + exchange['Open'] - timedelta(hours=1) + offset, symbol)

    prior_day = day_start
    day_start = day_start + timedelta(days=1)
  ##%%

    df_1m = day_candles
    df_5m = df_1m.resample('5min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    df_30m = df_1m.resample('30min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))

    df_5m['VWAP3'] = (df_5m['c']+df_5m['h']+df_5m['l'])/3
    df_5m['9EMA'] = df_5m['c'].ewm(span=9, adjust=False).mean()
    # Calculate the Adaptive Moving Average (AMA / KAMA)
    df_5m['AMA'] = utils.indicators.adaptive_moving_average(df_5m['c'], period=10, fast=2, slow=30)
    df_5m['AMA_VWAP'] = utils.indicators.adaptive_moving_average(df_5m['VWAP3'], period=10, fast=2, slow=30)
    df_5m['9EMA_VWAP'] = df_5m['VWAP3'].ewm(span=9, adjust=False).mean()

    # Define a rolling window size (e.g., 10 days)
    window_size = 3

    # Function to calculate the slope for a window
    def calculate_slope(x):
      indices = np.arange(len(x))  # Create a simple index for regression (0, 1, 2, ...)
      slope, intercept, r_value, p_value, std_err = linregress(indices, x)
      return slope

    def trend_estimation(h_1, l_1, h, l):
      # new lows and an overlap of at most 75 pct
      atr = (h_1 - l_1)
      if h_1 > h and l_1 > l and (atr - (h - l_1))/atr > 0.1:
        return -1
      elif h_1 < h and l_1 < l and (atr - (h_1 - l))/atr > 0.1:
        return 1
      else:
        return 0


    # Apply a rolling window to calculate the slope
    df_5m["SL_9EMA"] = df_5m["9EMA"].rolling(window=window_size).apply(calculate_slope, raw=True)
    df_5m["SL_AMA"] = df_5m["AMA"].rolling(window=window_size).apply(calculate_slope, raw=True)
    df_5m["SL_9EMA_VWAP"] = df_5m["9EMA_VWAP"].rolling(window=window_size).apply(calculate_slope, raw=True)
    df_5m["SL_AMA_VWAP"] = df_5m["AMA_VWAP"].rolling(window=window_size).apply(calculate_slope, raw=True)
    # df_5m['TREND'] = df_5m.rolling(window=2).apply(lambda x: trend_estimation(x.iloc[0]['h'], x.iloc[0].l, x.iloc[-1].h, x.iloc[-1].l))
    # df_5m['TREND'] = df_5m.rolling(2).apply(lambda x: trend_estimation(x.iloc[0]['h'], x.iloc[0]['l'], x.iloc[1]['h'], x.iloc[1]['l']))
    #%%
    micro_trend = 0
    macro_trend = 0
    micro_trend_series = np.zeros(len(df_5m), dtype=int)
    macro_trend_series = np.zeros(len(df_5m), dtype=int)
    macro_low = df_5m.iloc[0]['l']
    macro_high = df_5m.iloc[0]['h']
    last_macro_low_id = 0
    last_macro_high_id = 0
    i = 1
    backtracked = []
    while i < len(df_5m):
    # for i in range(1, len(df_5m)):
      micro_trend = trend_estimation(df_5m.iloc[i-1]['h'], df_5m.iloc[i-1]['l'], df_5m.iloc[i]['h'], df_5m.iloc[i]['l'])
      micro_trend_series[i] = micro_trend
      # macro trend not yet decided or micro trend in the same direction
      if micro_trend == macro_trend or macro_trend == 0 or micro_trend == 0:
        last_macro_low_id = last_macro_low_id if macro_low < df_5m.iloc[i]['l'] else i
        last_macro_high_id = last_macro_high_id if macro_high > df_5m.iloc[i]['h'] else i
        macro_high = max(macro_high, df_5m.iloc[i]['h'])
        macro_low = min(macro_low, df_5m.iloc[i]['l'])
        macro_trend = micro_trend if micro_trend != 0 else macro_trend
      # micro trend changed the direction
      elif micro_trend == -1*macro_trend:
        # see how far the micro direction has changed against the macro trend
        macro_trend_change = False
        macro_atr = macro_high - macro_low
        if macro_trend == -1:
          changed_atr = (df_5m.iloc[i]['h'] - macro_low)
          macro_trend_change = changed_atr / macro_atr > 0.9
        else: #macro_trend == 1:
          changed_atr = (macro_high - df_5m.iloc[i]['l'])
          macro_trend_change = changed_atr / macro_atr > 0.9

        print(f'{i} macro_atr={macro_atr:.2f} changed_atr={changed_atr:.2f} macro_trend_change {macro_trend_change}')
        # traceback to the last macro extrema to follow the opposite trend
        if macro_trend_change:
          last_extrema_id = max(last_macro_low_id, last_macro_high_id)

          if last_extrema_id in backtracked:
            print('infinite loop detected')
            break
          macro_high = df_5m.iloc[last_extrema_id]['h']
          macro_low = df_5m.iloc[last_extrema_id]['l']
          i = last_extrema_id
          macro_trend = 0
          print(f'back to {i}')
          backtracked.append(i)

      macro_trend_series[i] = macro_trend
      print(f'{i}: macro {macro_trend} micro {micro_trend}  [{macro_low} {macro_high}] [{df_5m.iloc[i].l} {df_5m.iloc[i].h}] [{last_macro_low_id} {last_macro_high_id}]')
      i = i + 1

    df_5m['MICRO_TREND'] = micro_trend_series
    df_5m['MACRO_TREND'] = macro_trend_series

    # # df_5m["SL_9EMA_VWAP_PEAKS"] = df_5m["SL_9EMA_VWAP"].copy().where(df_5m["SL_9EMA_VWAP"].abs() > 4, 0) + df_5m["SL_AMA_VWAP"].copy().where(df_5m["SL_AMA_VWAP"].abs() > 4, 0)
    # signal.find_peaks(df_5m["SL_9EMA_VWAP_PEAKS"], distance=3, prominence=4)
    #%%
    try:
      ## %%
      fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)

      date_str = day_start.strftime('%Y-%m-%d')
      ax1 = fig.add_subplot(2,1,1)
      ax2 = fig.add_subplot(2,1,2)

      overnight_h = overnight_candle.h.iat[0] if overnight_candle is not None else np.nan
      overnight_l = overnight_candle.l.iat[0] if overnight_candle is not None else np.nan

      indicator_hlines = [prior_day_candle.c.iat[0], prior_day_candle.h.iat[0], prior_day_candle.l.iat[0], overnight_h, overnight_l]
      fig.suptitle(f'{symbol} {date_str} 1m/5m/30m PriorDay: H {prior_day_candle.h.iat[0]:.2f}  C {prior_day_candle.c.iat[0]:.2f} L {prior_day_candle.l.iat[0]:.2f} On: H {overnight_h:.2f} L {overnight_l:.2f}')

      hlines=dict(hlines=indicator_hlines, colors=['#bf42f5'], linewidths=[0.5, 1, 1, 0.5, 0.5], linestyle=['--', *['-']*(len(indicator_hlines)-1)])

      # ema_plot = mpf.make_addplot(df_5m['9EMA'], ax=ax1, width=0.5, color="turquoise")
      ema_vwap_plot = mpf.make_addplot(df_5m['9EMA_VWAP'], ax=ax1, width=1, color="darkturquoise")
      # ama_plot = mpf.make_addplot(df_5m['AMA'], ax=ax1, width=0.5, color='gold')
      ama_vwap_plot = mpf.make_addplot(df_5m['AMA_VWAP'], ax=ax1, width=1, color='goldenrod')
      vwap3_plot = mpf.make_addplot(df_5m['VWAP3'], ax=ax1, width=1, color='darkgoldenrod')
      addplots = [vwap3_plot, ama_vwap_plot, ema_vwap_plot]

      mpf.plot(df_5m, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=addplots)

      # ema_slope_plot = mpf.make_addplot(df_5m['SL_9EMA'], ax=ax2, width=1, color="turquoise")
      # ama_slope_plot = mpf.make_addplot(df_5m['SL_AMA'], ax=ax2, width=1, color='gold')
      ama_vwap_slope_plot = mpf.make_addplot(df_5m['SL_AMA_VWAP'], ax=ax2, width=1, color='goldenrod')
      ema_vwap_slope_plot = mpf.make_addplot(df_5m['SL_9EMA_VWAP'], ax=ax2, width=1, color="darkturquoise")
      micro_trend_plot = mpf.make_addplot(df_5m['MICRO_TREND'], ax=ax2, width=1, color="gainsboro")
      macro_trend_plot= mpf.make_addplot(df_5m['MACRO_TREND'], ax=ax2, width=1, color="gray", linestyle='--')
      slope_df = df_5m.copy()
      slope_addplots = [ama_vwap_slope_plot, ema_vwap_slope_plot, micro_trend_plot, macro_trend_plot]

      mpf.plot(slope_df, type='line', ax=ax2,columns=['SL_AMA_VWAP']*5,  xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=slope_addplots)

      plt.show()
      ## %%
      # plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
      # plt.close()
      print(f'{symbol} finished {date_str}')
      ##%%
    except Exception as e:
      print(f'{symbol} error: {e}')
      # continue

#%%
def trend_estimation(h_1, l_1, h, l):
  # new lows and an overlap of at most 75 pct
  atr = (h_1 - l_1)
  if h_1 > h and l_1 > l and (atr - (h - l_1))/atr > 0.25:
    return -1
  elif h_1 < h and l_1 < l and (atr - (h_1 - l))/atr > 0.25:
    return 1
  else:
    return 0

h_1 = 16076.91
l_1 = 16031.36
h = 16052.88
l = 16026.86
print(trend_estimation( 16076.91,16031.36, 16052.88,16026.86 ))

#%%

# Expecting ohcl & vwap
def candle_following(series):
  macro_trend = 0
  micro_trend = 0
  for i in range(2, len(series)):
    if series.vwap.iloc[i-1] > series.vwap.iloc[i]:



