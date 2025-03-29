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
from finance.behavior_eval.change_evaluation import loc_10m

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%% get influx data
influx_client_df, influx_client = utils.influx.get_influx_clients()

symbols = ['IBDE40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100']
symbol = symbols[0]
# Create a directory
directory = f'N:/My Drive/Projects/Trading/Research/Plots/{symbol}_mpf_1m_5m_30m'
os.makedirs(directory, exist_ok=True)

# Create a directory
symbol_def = utils.influx.SYMBOLS[symbol]

exchange = symbol_def['EX']
tz = exchange['TZ']

dfs_ref_range = []
dfs_closing = []
first_day = tz.localize(dateutil.parser.parse('2020-01-02T00:00:00'))
last_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
day_start = first_day + timedelta(days=1)

#%%

# while day_start < last_day:
  prior_day = day_start - timedelta(days=1)
  prior_day_candle = utils.influx.get_candles_range_aggregate(prior_day + exchange['Open'], prior_day + exchange['Close'], symbol)
  overnight_candle= utils.influx.get_candles_range_aggregate(prior_day + exchange['Close'], day_start + exchange['Open'] - timedelta(hours=1), symbol)

  day_end = day_start + symbol_def['EX']['Close'] + timedelta(minutes=1)
  # get the following data for daily assignment
  day_candles = utils.influx.get_candles_range_raw(day_start+exchange['Open']-timedelta(hours=1, minutes=30), day_end, symbol)
  day_start = day_end
  if symbol not in day_candles:
    print(f'no data for {day_start.isoformat()}')
    # continue
##%%

  df_1m = day_candles
  # df_2m = df_raw.resample('2min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  df_5m = df_1m.resample('5min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  # df_10m = df_raw.resample('10min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  # df_15m = df_raw.resample('15min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  df_30m = df_1m.resample('30min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  # df_60m = df_raw.resample('60min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))

  # df_2m.index = df_2m.index + pd.DateOffset(minutes=-1)
  # df_10m.index = df_10m.index + pd.DateOffset(minutes=-1)
  # df_60m.index = df_60m.index + pd.DateOffset(minutes=-1)
  # df_10m.index = df_10m.index + pd.DateOffset(minutes=5)
  # df_60m.index = df_60m.index + pd.DateOffset(minutes=30)

#%%
  # try:
    plt.close()
    fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)

    date_str = day_start.strftime('%Y-%m-%d')
    fig.suptitle(f'{symbol} {date_str} 1m/5m/30m')
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    prior_close = mpf.make_addplot([] * len(df), color="red", linestyle="--", linewidth=1.5)
    prior_close = mpf.make_addplot([] * len(df), color="red", linestyle="--", linewidth=1.5)


    mpf.plot(df_1m, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING,  xrotation=0, datetime_format='%H:%M', tight_layout=True)
    mpf.plot(df_5m.iloc[:-1], type='candle', ax=ax2, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35))
    mpf.plot(df_30m.iloc[:-1], type='candle', ax=ax3, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35))

    ticks = pd.date_range(df_1m.index.min(),df_1m.index.max()+timedelta(minutes=1),freq='30min')
    ticklabels = [ tick.time().strftime('%H:%M') for tick in ticks ]
    loc_1m = [df_1m.index.get_loc(tick) for tick in ticks]
    ax1.set_xticks(loc_1m)
    ax1.set_xticklabels(ticklabels)
    ax1.set_xlim(-2.5, loc_1m[-1] + 2.5)

    ticklabels = [ (tick+timedelta(minutes=2.5)).strftime('%H:%M') for tick in ticks ]
    loc_5m = [df_5m.index.get_loc(tick) for tick in ticks]
    ax2.set_xticks(loc_5m)
    ax2.set_xticklabels(ticklabels)
    ax2.set_xlim(-1, loc_5m[-1] )

    ticks = pd.date_range(df_1m.index.min(),df_1m.index.max()+timedelta(minutes=1),freq='30min')[:-1]#+pd.DateOffset(minutes=15)
    ticklabels = [ (tick+timedelta(minutes=15)).time().strftime('%H:%M') for tick in ticks ]
    loc_30m = [df_30m.index.get_loc(tick) for tick in ticks]
    ax3.set_xticks([ df_30m.index.get_loc(tick) for tick in ticks ])
    ax3.set_xticklabels(ticklabels)
    ax3.set_xlim(-0.583, loc_30m[-1]+0.583)


    # plt.savefig(f'finance/_data/dax_plots_mpl/IBDE40_{date_str}.png', dpi=300, bbox_inches='tight')  # High-quality save
    # plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
    # plt.savefig(f'N:/My Drive/Projects/Trading/Research/DAX/IBDE40_mpf_2m_10m_60m/IBDE40_{date_str}.svg', bbox_inches='tight')  # High-quality save
    # plt.savefig(f'N:/My Drive/Projects/Trading/Research/DAX/IBDE40_mpf_2m_10m_60m/IBDE40_{date_str}.jpg', bbox_inches='tight')  # High-quality save
    # plt.close()
    plt.show()
    print(f'finished {date_str}')
  # except Exception as e:
  #   print(f'error: {e}')
  #   continue

