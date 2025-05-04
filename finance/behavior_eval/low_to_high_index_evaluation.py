#%%
import pickle
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
  ##%%
  # Create a directory
  directory = f'N:/My Drive/Projects/Trading/Research/Plots/swing/{symbol}_low_to_high_trend_eval'
  os.makedirs(directory, exist_ok=True)

  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]

  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = tz.localize(dateutil.parser.parse('2020-01-09T00:00:00'))

  now = datetime.now(tz)
  last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  offset = timedelta(hours=0)
  ##%%
  while day_start < last_day:
    ##%%
    day_end = day_start + exchange['Close'] + timedelta(hours=1) + offset
    # get the following data for daily assignment
    day_candles = utils.influx.get_candles_range_aggregate(day_start+exchange['Open']-timedelta(hours=1, minutes=0)+offset, day_end, symbol, '5m')
    if day_candles is None or len(day_candles) < 20:
      day_start = day_start + timedelta(days=1)
      print(f'{symbol} no data for {day_start.isoformat()}')
      # raise Exception('no data')
      continue
    last_saturday = utils.time.get_last_saturday(day_start)
    current_week_candle = utils.influx.get_candles_range_aggregate(last_saturday, prior_day + exchange['Close'] + offset, symbol)
    prior_week_candle = utils.influx.get_candles_range_aggregate(last_saturday-timedelta(days=7), last_saturday, symbol)
    prior_day_candle = utils.influx.get_candles_range_aggregate(prior_day + exchange['Open'] + offset, prior_day + exchange['Close'] + offset, symbol)
    overnight_candle= utils.influx.get_candles_range_aggregate(day_start + offset, day_start + exchange['Open'] - timedelta(hours=1) + offset, symbol)

    prior_day = day_start
    day_start = day_start + timedelta(days=1)
  ##%%

    # Extrema algorithm
    # 1. Start with the o, h, c, l of the current day
    # 2. Determine if time_h is before or after time_l
    # 3. Assume time_h > time_l for the following steps
    # 4. Determine if time_h == time_o as isOpenHigh
    # 5. Determine if time_l == time_c as isCloseLow
    # 6. If not isOpenHigh
    # 7.
    # extrema = {"ts": , "type": 'h' | 'l',  'dpdh': , 'dpdl' , 'dcwh' , 'dcwl' , 'dpwh' , 'dpwl' }

    pdh = prior_day_candle.h.iat[0]
    pdl = prior_day_candle.l.iat[0]
    cwh = current_week_candle.h.iat[0] if current_week_candle is not None else np.nan
    cwl = current_week_candle.l.iat[0] if current_week_candle is not None else np.nan
    pwh = prior_week_candle.h.iat[0]
    pwl = prior_week_candle.l.iat[0]
    onh = overnight_candle.h.iat[0] if overnight_candle is not None else np.nan
    onl = overnight_candle.l.iat[0] if overnight_candle is not None else np.nan

    df_5m = day_candles
    start = df_5m.iloc[0]
    low = df_5m['l'].min()
    extreme_low = {"ts": df_5m['l'].idxmin(), "type": 'l', "value": low, "dpdh": pdh-low, "dpdl": pdl-low,
                   "dcwh": cwh-low, "dcwl": cwl-low, "dpwh": pwh-low, "dpwl": pwl-low, "donh": onh-low, "donl": onl-low}
    high = df_5m['h'].max()
    extreme_high = {"ts": df_5m['h'].idxmax(), "type": 'h', "value":high, "dpdh": pdh-high, "dpdl": pdl-high,
                    "dcwh": cwh-high, "dcwl": cwl-low, "dpwh": pwh-high, "dpwl": pwl-high, "donh": onh-high, "donl": onl-high}
    end = df_5m.iloc[-1]

    is_up_move_day = extreme_low['ts'] < extreme_high['ts']
    extrema_lh = [extreme_low, extreme_high] if is_up_move_day else [extreme_high, extreme_low]

    is_open_extreme = extrema_lh[0]['ts'] == start.name
    is_close_extreme = extrema_lh[-1]['ts'] == end.name

    ##%%
    def get_extrema(df, is_next_high, begin, stop):
      current_direction = -1 if is_next_high else 1
      current = begin
      extrema = []
      while current < stop if begin < stop else current > stop:
        slice_index = (df.index > current) & (df.index <= stop) if begin < stop else (df.index < current) & (df.index >= stop)
        if not any(slice_index):
          return extrema
        if current_direction < 0:
          next_extrema = df[slice_index]['l'].idxmin()
          value = df[slice_index]['l'].min()
        else:
          next_extrema = df[slice_index]['h'].idxmax()
          value = df[slice_index]['h'].max()

        extrema.append({"ts": next_extrema, "type": 'h' if current_direction > 0 else 'l', "value": value,
                        "dpdh": pdh-value, "dpdl": pdl-value, "dcwh": cwh-value, "dcwl": cwl-value, "dpwh": pwh-value,
                        "dpwl": pwl-value, "donh": onh-value, "donl": onl-value})
        current = next_extrema
        current_direction = -1*current_direction
      return extrema

    ##%%
    start_extrema = [] if is_open_extreme else get_extrema(df_5m, ~is_up_move_day, extrema_lh[0]['ts'], start.name)
    start_extrema.reverse()
    end_extrema = [] if is_close_extreme else get_extrema(df_5m, is_up_move_day, extrema_lh[-1]['ts'], end.name)
    extrema_ordered = [*start_extrema, *extrema_lh, *end_extrema]
    # time_diffs = np.diff(extrema_ordered)

    # Follow algorithm
    # 1. Start with a candle range defined as [start end] whereas start['l'] is the lowest value and end['h'] is the highest value
    # 2. Follow the lows of the candles as long as the last candle low C_1l is <= current candle low C_l continue
    # 3. If C_l < C_1l follow the pullback until C_l > C_1l again.
    #    a. Determine the duration of the pullback as well as the difference from the first pullback candle high to the last pullback candle low
    #       and the first pullback candle opens to the last pullback candle close.

    ##%%
    # SC = SingleCandle in the opposite direction
    # PB = Pullback that breaks the continuation of the previous candle following
    # pullback = {"start": , "end": , "duration": , "h": , "l": , "o": , "c": , "type": 'SC', "direction": 1 | -1"}"
    ##%%
    pullbacks = []
    for i in range(1, len(extrema_ordered)):
      start_extrema = extrema_ordered[i-1]
      end_extrema = extrema_ordered[i]
      direction = 1 if start_extrema['type'] == 'l' else -1
      df_dir = df_5m[(df_5m.index > start_extrema['ts']) & (df_5m.index < end_extrema['ts'])]
      current_pullback = None
      for j in range(1, len(df_dir)):
        if direction > 0:
          is_continuation = df_dir.iloc[j-1]['l'] <= df_dir.iloc[j]['l']
        else:
          is_continuation = df_dir.iloc[j-1]['h'] >= df_dir.iloc[j]['h']
        if is_continuation and current_pullback is not None:
          current_pullback['end'] = df_dir.iloc[j-1].name
          if direction > 0:
            current_pullback['l'] = df_dir.iloc[j-1]['l']
          else:
            current_pullback['h'] = df_dir.iloc[j-1]['h']
          current_pullback['c'] = df_dir.iloc[j-1]['c']
          pullbacks.append(current_pullback)
          current_pullback = None

        is_positive_candle = df_dir.iloc[j]['o'] <= df_dir.iloc[j]['c']
        if is_continuation and ((direction > 0 and not is_positive_candle) or (direction < 0 and is_positive_candle)):
          pullbacks.append({"start": df_dir.iloc[j-1].name, "end": df_dir.iloc[j].name, "h": df_dir.iloc[j]['h'],
                            "l": df_dir.iloc[j]['l'], "o": df_dir.iloc[j]['o'], "c": df_dir.iloc[j]['c'], "type": 'SC', "direction": direction})

        if not is_continuation and current_pullback is None:
          current_pullback = {"start": df_dir.iloc[j-1].name, "o": df_dir.iloc[j]['o'], "type": 'PB', "direction": direction}
          if direction > 0:
            current_pullback['h'] = df_dir.iloc[j]['h']
          else:
            current_pullback['l'] = df_dir.iloc[j]['l']

    ##%%

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

    # Apply a rolling window to calculate the slope
    # df_5m["SL_9EMA"] = df_5m["9EMA"].rolling(window=window_size).apply(calculate_slope, raw=True)
    # df_5m["SL_AMA"] = df_5m["AMA"].rolling(window=window_size).apply(calculate_slope, raw=True)
    # df_5m["SL_9EMA_VWAP"] = df_5m["9EMA_VWAP"].rolling(window=window_size).apply(calculate_slope, raw=True)
    # df_5m["SL_AMA_VWAP"] = df_5m["AMA_VWAP"].rolling(window=window_size).apply(calculate_slope, raw=True)

    ##%%
    try:
      ## %%
      fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)
      ax1 = fig.add_subplot(1,1,1)

      date_str = day_start.strftime('%Y-%m-%d')

      overnight_h = overnight_candle.h.iat[0] if overnight_candle is not None else np.nan
      overnight_l = overnight_candle.l.iat[0] if overnight_candle is not None else np.nan

      indicator_hlines = [prior_day_candle.c.iat[0], prior_day_candle.h.iat[0], prior_day_candle.l.iat[0], overnight_h, overnight_l, cwl, cwh, prior_week_candle.l.iat[0], prior_week_candle.h.iat[0]]
      fig.suptitle(f'{symbol} {date_str} 5m O {df_5m.o.iat[0]:.2f} H {df_5m.h.iat[0]:.2f} C {df_5m.c.iat[0]:.2f} L {df_5m.l.iat[0]:.2f} \n' +
                   f'PriorDay: H {prior_day_candle.h.iat[0]:.2f} C {prior_day_candle.c.iat[0]:.2f} L {prior_day_candle.l.iat[0]:.2f}  On: H {overnight_h:.2f} L {overnight_l:.2f} \n' +
                   f'CurrentWeek: H {cwh:.2f} L {cwl:.2f}  PriorWeek: H {prior_week_candle.h.iat[0]:.2f} L {prior_week_candle.l.iat[0]:.2f}')

      hlines=dict(hlines=indicator_hlines, colors=[*['#bf42f5']*5, *['#3179f5']*4], linewidths=[0.5, 1, 1, 0.5, 0.5], linestyle=['--', *['-']*(len(indicator_hlines)-1)])

      extrema_aline= []
      for extrema in extrema_ordered:
        extrema_aline.append((extrema['ts'], extrema['value']))

      pb_alines= []
      sc_alines= []
      for pullback in pullbacks:
        if pullback['type'] == 'PB':
          pb_alines.append([(pullback['start'], pullback['l']), (pullback['end'], pullback['l'])])
          pb_alines.append([(pullback['start'], pullback['h']), (pullback['end'], pullback['h'])])
        else:
          sc_alines.append([(pullback['start'], pullback['l']), (pullback['end'], pullback['l'])])
          sc_alines.append([(pullback['start'], pullback['h']), (pullback['end'], pullback['h'])])

      alines=dict(alines=[extrema_aline, *pb_alines, *sc_alines], colors=['black', *['darkblue'] * len(pb_alines), *['darkred'] * len(sc_alines)], alpha=0.5, linewidths=[0.25], linestyle=['--', *['-'] * len(pb_alines)])

      ema_plot = mpf.make_addplot(df_5m['9EMA'], ax=ax1, width=0.5, color="turquoise")
      # ema_vwap_plot = mpf.make_addplot(df_5m['9EMA_VWAP'], ax=ax1, width=0.5, color="darkturquoise")
      ama_plot = mpf.make_addplot(df_5m['AMA'], ax=ax1, width=0.5, color='goldenrod')
      # ama_vwap_plot = mpf.make_addplot(df_5m['AMA_VWAP'], ax=ax1, width=0.5, color='darkgoldenrod')
      vwap3_plot = mpf.make_addplot(df_5m['VWAP3'], ax=ax1, width=1, color='gold')
      addplots = [vwap3_plot, ama_plot, ema_plot]

      mpf.plot(df_5m, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING, alines=alines, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=addplots)

      # Use MaxNLocator to increase the number of ticks
      ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=15))  # Increase number of ticks on x-axis
      ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))  # Increase number of ticks on y-axis

      plt.show()
      ## %%
      metadata = {"pullbacks": pullbacks, "extrema": extrema_ordered, "VWAP": df_5m['VWAP3']}
      with open(f'{directory}/{symbol}_{date_str}.pkl', "wb") as f:
        pickle.dump(metadata, f)
      ## %%
      plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
      # plt.close()
      print(f'{symbol} finished {date_str}')
      ##%%
    except Exception as e:
      print(f'{symbol} error: {e}')
      continue

