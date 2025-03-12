# %%
import datetime

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

from finance.utils.pct import percentage_change
import finplot as fplt
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

from finance.behavior_eval.influx_utils import get_candles_range_aggregate_query, get_candles_range_raw_query

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

##%% get influx data
DB_INDEX = 'index'
DB_CFD = 'cfd'
DB_FOREX = 'forex'

influx_client_df = idb.DataFrameClient()
influx_client = idb.InfluxDBClient()

indices = influx_client.query('show measurements', database=DB_INDEX)
cfds = influx_client.query('show measurements', database=DB_CFD)
forex = influx_client.query('show measurements', database=DB_FOREX)

get_values = lambda x: [y[0] for y in x.raw['series'][0]['values']]
print('Indices: ', get_values(indices))
print('Cfds: ', get_values(cfds))
print('Forex: ', get_values(forex))


# %%
directory = f'N:/My Drive/Projects/Trading/Research/Plots/thu_fri_mon'
os.makedirs(directory, exist_ok=True)
# symbol = 'IBDE40'
symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
           *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]

for symbol, tz in symbols:
# tz = pytz.timezone('Europe/Berlin')
# tz = pytz.timezone('Europe/London')
# Create a directory
  directory = f'N:/My Drive/Projects/Trading/Research/Plots/{symbol}_mpf_2m_10m_60m'
  os.makedirs(directory, exist_ok=True)

  dfs_ref_range = []
  dfs_closing = []
  first_day = tz.localize(dateutil.parser.parse('2020-01-02T00:00:00'))
  last_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
  day_start = first_day
  prior_close = None
  df_raw = None
  ## %%
  while day_start < last_day:
    if df_raw is not None:
      prior_close = df_raw.iloc[-1].c
    day_end = day_start + datetime.timedelta(days=1)
    # get the following data for daily assignment
    day_candles = influx_client_df.query(get_candles_range_raw_query(day_start, day_end, symbol), database=DB_CFD)
    day_start = day_end
    if symbol not in day_candles:
      print(f'no data for {day_start.isoformat()}')
      continue
    ##%%
    df_raw = day_candles[symbol].tz_convert(tz)
    #filter everything before 3:00 and after 22:00
    df_raw = df_raw[(df_raw.index.time >= datetime.time(3, 0)) & (df_raw.index.time <= datetime.time(22, 0))]
    df_2m = df_raw.resample('2min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    # df_5m = df_raw.resample('5min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    df_10m = df_raw.resample('10min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    # df_15m = df_raw.resample('15min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    # df_30m = df_raw.resample('30min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    df_60m = df_raw.resample('60min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))

    df_10m.index = df_10m.index + pd.DateOffset(minutes=5)
    df_60m.index = df_60m.index + pd.DateOffset(minutes=30)

    # Map the custom column names to the required OHLC column names
    column_mapping = list({
                            'Open': 'o',  # Map "Open" to our custom "Start" column
                            'High': 'h',  # Map "High" to "Highest"
                            'Low': 'l',  # Map "Low" to "Lowest"
                            'Close': 'c',  # Map "Close" to "Ending"
                            'Volume': 'v'  # Map "Volume" to "Volume_Traded"
                          }.values())
    ##%%
    try:
      fig = mpf.figure(style='yahoo', figsize=(20, 12), tight_layout=True)

      date_str = day_start.strftime('%Y-%m-%d')
      prior_close_str = f'Prior Close: {prior_close:.2f}' if prior_close is not None else 'N/A'
      fig.suptitle(f'{symbol} {date_str} 2m/10m/60m {prior_close_str}')
      ax1 = fig.add_subplot(3, 1, 1)
      ax2 = fig.add_subplot(3, 1, 2)
      ax3 = fig.add_subplot(3, 1, 3)

      mpf.plot(df_2m, type='candle', ax=ax1, columns=column_mapping, xrotation=0, datetime_format='%H:%M',
               tight_layout=True)
      mpf.plot(df_10m, type='candle', ax=ax2, columns=column_mapping, xrotation=0, datetime_format='%H:%M',
               tight_layout=True, scale_width_adjustment=dict(candle=1.35))
      mpf.plot(df_60m, type='candle', ax=ax3, columns=column_mapping, xrotation=0, datetime_format='%H:%M',
               tight_layout=True, scale_width_adjustment=dict(candle=1.35))

      ticks = pd.date_range(df_2m.index.min(), df_2m.index.max(), freq='30min').ceil('30min')[0:-1]
      ticklabels = [tick.time().strftime('%H:%M') for tick in ticks]
      loc_2m = [df_2m.index.get_loc(tick) for tick in ticks]
      ax1.set_xticks(loc_2m)
      ax1.set_xticklabels(ticklabels)
      ax1.set_xlim(-2.5, loc_2m[-1] + 15)

      ticks = pd.date_range(df_2m.index.min(), df_2m.index.max(), freq='30min').ceil('30min')[0:-1] + pd.Timedelta(
        minutes=15)
      ticklabels = [tick.time().strftime('%H:%M') for tick in ticks]
      loc_10m = [df_10m.index.get_loc(tick) for tick in ticks]
      ax2.set_xticks(loc_10m)
      ax2.set_xticklabels(ticklabels)
      ax2.set_xlim(-1, loc_10m[-1] + 1.5)

      ticks = pd.date_range(df_2m.index.min(), df_2m.index.max(), freq='1h').floor('1h')[0:-1] + pd.DateOffset(minutes=30)
      ticklabels = [tick.time().strftime('%H:%M') for tick in ticks]
      loc_60m = [df_60m.index.get_loc(tick) for tick in ticks]
      ax3.set_xticks([df_60m.index.get_loc(tick) for tick in ticks])
      ax3.set_xticklabels(ticklabels)
      ax3.set_xlim(-0.583, loc_60m[-1] + 0.5)

      if prior_close is not None:
        ax1.axhline(y=prior_close, color="black", linestyle="--", linewidth=1, label="Prior Close")
        ax2.axhline(y=prior_close, color="black", linestyle="--", linewidth=1, label="Prior Close")
        ax3.axhline(y=prior_close, color="black", linestyle="--", linewidth=1, label="Prior Close")

      plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
      # plt.savefig(f'finance/_data/dax_plots_mpl/IBDE40_{date_str}.png', dpi=300, bbox_inches='tight')  # High-quality save
      # plt.show()
      plt.close()
      print(f'finished {symbol} {date_str}')
    except Exception as e:
      print(f'error: {e}')
      plt.close()
      # continue



