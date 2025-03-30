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

import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%% get influx data
influx_client_df, influx_client = utils.influx.get_influx_clients()

#%%
symbols = ['IBDE40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100']
# symbol = symbols[3]
for symbol in symbols:
  # Create a directory
  directory = f'N:/My Drive/Projects/Trading/Research/Plots/{symbol}_mpf_1m_5m_30m'
  os.makedirs(directory, exist_ok=True)

  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]

  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = tz.localize(dateutil.parser.parse('2022-01-03T00:00:00'))
  last_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  ##%%
  while day_start < last_day:
    day_end = day_start + exchange['Close'] + timedelta(hours=1)
    # get the following data for daily assignment
    day_candles = utils.influx.get_candles_range_raw(day_start+exchange['Open']-timedelta(hours=1, minutes=0), day_end, symbol)
    if day_candles is None or len(day_candles) == 0:
      day_start = day_start + timedelta(days=1)
      print(f'no data for {day_start.isoformat()}')
      # raise Exception('no data')
      continue
    prior_day_candle = utils.influx.get_candles_range_aggregate(prior_day + exchange['Open'], prior_day + exchange['Close'], symbol)
    overnight_candle= utils.influx.get_candles_range_aggregate(day_start, day_start + exchange['Open'] - timedelta(hours=1), symbol)

    prior_day = day_start
    day_start = day_start + timedelta(days=1)
  ##%%

    df_1m = day_candles
    # df_2m = df_raw.resample('2min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    df_5m = df_1m.resample('5min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    # df_10m = df_raw.resample('10min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    # df_15m = df_raw.resample('15min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    df_30m = df_1m.resample('30min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    # df_60m = df_raw.resample('60min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))

    df_1m['9EMA'] = df_1m['c'].ewm(span=9, adjust=False).mean()
    df_5m['9EMA'] = df_5m['c'].ewm(span=9, adjust=False).mean()

    # Calculate the Adaptive Moving Average (AMA / KAMA)
    df_5m['AMA'] = utils.indicators.adaptive_moving_average(df_5m['c'], period=10, fast=2, slow=30)

  ##%%
    try:
      fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)

      date_str = day_start.strftime('%Y-%m-%d')
      ax1 = fig.add_subplot(3,1,1)
      ax2 = fig.add_subplot(3,1,2)
      ax3 = fig.add_subplot(3,1,3)

      overnight_h = overnight_candle.h.iat[0] if overnight_candle is not None else np.nan
      overnight_l = overnight_candle.l.iat[0] if overnight_candle is not None else np.nan

      indicator_hlines = [prior_day_candle.c.iat[0], prior_day_candle.h.iat[0], prior_day_candle.l.iat[0], overnight_h, overnight_l]
      fig.suptitle(f'{symbol} {date_str} 1m/5m/30m PriorDay: H {prior_day_candle.h.iat[0]:.2f}  C {prior_day_candle.c.iat[0]:.2f} L {prior_day_candle.l.iat[0]:.2f} On: H {overnight_h:.2f} L {overnight_l:.2f}')

      hlines=dict(hlines=indicator_hlines, colors=['#bf42f5'], linewidths=[0.5, 0.5, 0.5, 1, 1], linestyle=['--', *['-']*(len(indicator_hlines)-1)])

      # ema_plot = mpf.make_addplot( df_1m['9EMA'], ax=ax1, color="yellow")
      ama_plot = mpf.make_addplot(df_5m['AMA'], ax=ax2, width=0.5, color='gold')

      mpf.plot(df_1m, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING,  xrotation=0, datetime_format='%H:%M', tight_layout=True, hlines=hlines, warn_too_much_data=700)
      mpf.plot(df_5m, type='candle', ax=ax2, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=ama_plot)
      mpf.plot(df_30m, type='candle', ax=ax3, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35), hlines=hlines)

      ticks = pd.date_range(df_1m.index.min(),df_1m.index.max()+timedelta(minutes=1),freq='30min')
      ticklabels = [ tick.time().strftime('%H:%M') for tick in ticks ]
      loc_1m = [df_1m.index.get_loc(tick) for tick in ticks[:-1]]
      last_tick_1m = 2* loc_1m[-1] - loc_1m[-2]
      loc_1m.append(last_tick_1m)
      ax1.set_xticks(loc_1m)
      ax1.set_xticklabels(ticklabels)
      ax1.set_xlim(-2.5, last_tick_1m + 2.5)

      ticklabels = [ (tick+timedelta(minutes=2.5)).strftime('%H:%M') for tick in ticks ]
      loc_5m = [df_5m.index.get_loc(tick) for tick in ticks[:-1]]
      last_tick_5m = 2*loc_5m[-1] - loc_5m[-2]
      loc_5m.append(last_tick_5m)
      ax2.set_xticks(loc_5m)
      ax2.set_xticklabels(ticklabels)
      ax2.set_xlim(-1, last_tick_5m )

      ticklabels = [ (tick+timedelta(minutes=15)).time().strftime('%H:%M') for tick in ticks ]
      loc_30m = [df_30m.index.get_loc(tick) for tick in ticks[:-1]]
      last_tick_30m = 2*loc_30m[-1] - loc_30m[-2]
      loc_30m.append(last_tick_30m)
      ax3.set_xticks(loc_30m)
      ax3.set_xticklabels(ticklabels)
      ax3.set_xlim(-0.583, last_tick_30m - 0.45)

      plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
      plt.close()
      # plt.show()
      print(f'finished {date_str}')
    except Exception as e:
      print(f'error: {e}')
      continue

