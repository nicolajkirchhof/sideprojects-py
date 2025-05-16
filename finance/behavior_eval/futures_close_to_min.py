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

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
symbols = ['IBDE40', 'IBES35', 'IBFR40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225']
symbol = symbols[0]
for symbol in symbols:
  #%% Create a directory
  directory_evals = f'N:/My Drive/Projects/Trading/Research/Strategies/close_to_min/{symbol}'
  directory_plots = f'N:/My Drive/Projects/Trading/Research/Plots/close_to_min/{symbol}'
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
  df_5m_two_weeks, df_30m_two_weeks, df_day_two_weeks = utils.influx.get_5m_30m_day_date_range_with_indicators(day_start, day_start+timedelta(days=14), symbol)
  #%%
  while day_start < last_day:
    #%%
    if df_5m_two_weeks.index.max() < day_start + timedelta(days=5):
      df_5m_two_weeks, df_30m_two_weeks, df_day_two_weeks = utils.influx.get_5m_30m_day_date_range_with_indicators(day_start, day_start+timedelta(days=14), symbol)

    ##%%
    day_end = day_start + exchange['Close'] + timedelta(hours=1)
    # get the following data for daily assignment
    ##%%

    df_5m = df_5m_two_weeks[(df_5m_two_weeks.index >= day_start+exchange['Open']-timedelta(hours=1, minutes=0)) & (df_5m_two_weeks.index <= day_end)].copy()
    # df_30m = df_30m_two_weeks[(df_30m_two_weeks.index >= day_start+exchange['Open']-timedelta(hours=1, minutes=0)) & (df_30m_two_weeks.index <= day_end)].copy()
    # df_day = df_day_two_weeks[(df_day_two_weeks.index >= day_start - timedelta(days=14)) & (df_day_two_weeks.index <= day_start + timedelta(days=14))]

    if df_5m is None or len(df_5m) < 30:
      day_start = day_start + timedelta(days=1)
      print(f'{symbol} no data for {day_start.isoformat()}')
      # raise Exception('no data')
      continue
    last_saturday = utils.time.get_last_saturday(day_start)
    current_week_candle = df_5m_two_weeks[(df_5m_two_weeks.index >= last_saturday) & (df_5m_two_weeks.index <= prior_day + exchange['Close'])]
    prior_week_candle = df_5m_two_weeks[(df_5m_two_weeks.index >= last_saturday-timedelta(days=7)) & (df_5m_two_weeks.index <= last_saturday)]
    prior_day_candle = df_5m_two_weeks[(df_5m_two_weeks.index >= prior_day + exchange['Open']) & (df_5m_two_weeks.index <= prior_day + exchange['Close'])]
    overnight_candle = df_5m_two_weeks[(df_5m_two_weeks.index >= day_start) & (df_5m_two_weeks.index <= day_start + exchange['Open'] - timedelta(hours=1))]

    # Extrema algorithm
    # 1. Start with the o, h, c, l of the current day
    # 2. Determine if time_h is before or after time_l
    # 3. Assume time_h > time_l for the following steps
    # 4. Determine if time_h == time_o as isOpenHigh
    # 5. Determine if time_l == time_c as isCloseLow
    # 6. If not isOpenHigh
    # 7.
    # extrema = {"ts": , "type": 'h' | 'l',  'dpdh': , 'dpdl' , 'dcwh' , 'dcwl' , 'dpwh' , 'dpwl' }

    ##%%
    intraday_filter = (df_5m.index >= day_start + exchange['Open']) & (df_5m.index <= day_start + exchange['Close'])
    cdh = df_5m[intraday_filter].h.max()
    cdl = df_5m[intraday_filter].l.min()
    cdo = df_5m[intraday_filter].o.iat[0]
    cdc = df_5m[intraday_filter].c.iloc[-1]
    pdh = prior_day_candle.h.max() if not prior_day_candle.empty else np.nan
    pdl = prior_day_candle.l.min() if not prior_day_candle.empty else np.nan
    pdc = prior_day_candle.c.iat[-1] if not prior_day_candle.empty else np.nan
    cwh = current_week_candle.h.max() if not current_week_candle.empty else np.nan
    cwl = current_week_candle.l.min() if not current_week_candle.empty else np.nan
    pwh = prior_week_candle.h.max()
    pwl = prior_week_candle.l.min()
    onh = overnight_candle.h.max() if not overnight_candle.empty else np.nan
    onl = overnight_candle.l.min() if not overnight_candle.empty else np.nan
    #%%
    # find crossed candles for different max times

    all_dists = []
    min_time = day_start + exchange['Open']
    max_time = day_start + exchange['Open'] + timedelta(minutes=30)
    while max_time < day_start + exchange['Close'] - timedelta(hours=3):
      filter_max_time = (df_5m.index.time >= min_time.time()) & (df_5m.index.time <= max_time.time())
      if not any(filter_max_time):
        continue
      df_5m_filtered = df_5m[filter_max_time]
      closest_indices = df_5m_filtered[(df_5m_filtered.l < cdc) & (cdc < df_5m_filtered.h)].index.to_list()
      dists = [0]*len(closest_indices)
      if len(closest_indices) == 0:
        ##%%
        closest_min = (df_5m_filtered.l - cdc).abs().idxmin()
        closest_max = (df_5m_filtered.h - cdc).abs().idxmax()
        closest_indices = [max(closest_min, closest_max)]
        candle = df_5m_filtered.loc[closest_indices]
        dists = [min(abs(candle.l.iat[0] - cdc), abs(cdc - candle.h.iat[0]))]
      for ind, dist in zip(closest_indices, dists):
        all_dists.append({'ts': ind, 'date': ind.date(), 'weekday': ind.weekday(), 'time': ind.time(), 'dist': dist, 'minT': min_time.time(), 'maxT': max_time.time()})
      min_time = max_time
      max_time = min_time + timedelta(minutes=30)

  #%%
    date_str = day_start.strftime('%Y-%m-%d')
    # metadata = {"dists": all_dists, "day": day_start}
    # with open(f'{directory_evals}/{symbol}_{date_str}.pkl', "wb") as f:
    #   pickle.dump(metadata, f)
    df_all_dists = pd.DataFrame(all_dists)
    df_all_dists.to_pickle(f'{directory_evals}/{symbol}_{date_str}.pkl')
    print(f'{symbol} finished {date_str}')
    ##%%
    prior_day = day_start
    day_start = day_start + timedelta(days=1)

#%%
try:
#%%
# New setup
# |-------------------------|
# |           5m            |
# | ------------------------|
# |   D   |       30m       |
# | ------------------------|

  fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)
  gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 2])


  date_str = day_start.strftime('%Y-%m-%d')
  ax1 = fig.add_subplot(gs[0, :])
  ax2 = fig.add_subplot(gs[1, 0])
  ax3 = fig.add_subplot(gs[1, 1])

  indicator_hlines = [cdc, pdc, pdh, pdl, onh, onl, cwh, cwl, pwh, pwl]
  fig.suptitle(f'{symbol} {date_str} 5m O {cdo:.2f} H {cdh:.2f} C {cdc:.2f} L {cdl:.2f} \n' +
         f'PriorDay: H {pdh:.2f} C {pdc:.2f} L {pdl:.2f}  On: H {onh:.2f} L {onl:.2f} \n' +
         f'CurrentWeek: H {cwh:.2f} L {cwl:.2f}  PriorWeek: H {pwh:.2f} L {pwl:.2f}')

  hlines=dict(hlines=indicator_hlines, colors=['blue', *['#bf42f5']*5, *['#3179f5']*4], linewidths=[0.6]*4+[0.4]*6, linestyle=['-', '--', *['-']*(len(indicator_hlines)-1)])

  ind_5m_ema20_plot = mpf.make_addplot(df_5m['20EMA'], ax=ax1, width=0.6, color="#FF9900", linestyle='--')
  ind_5m_ema240_plot = mpf.make_addplot(df_5m['240EMA'], ax=ax1, width=0.6, color='#0099FF', linestyle='--')
  ind_vwap3_plot = mpf.make_addplot(df_5m['VWAP3'], ax=ax1, width=0.4, color='magenta')

  mpf.plot(df_5m, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True,
           scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=[ind_5m_ema20_plot, ind_5m_ema240_plot, ind_vwap3_plot])

  # Use MaxNLocator to increase the number of ticks
  ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))  # Increase number of ticks on x-axis
  ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))  # Increase number of ticks on y-axis

  plt.show()
    # %%
 except Exception as e:
  print(f'{symbol} error: {e}')
  # continue
