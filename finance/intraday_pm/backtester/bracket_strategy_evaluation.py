#%%
import datetime

import dateutil
import numpy as np
import scipy

import pandas as pd

import influxdb as idb

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from finance.utils import percentage_change
# import finplot as fplt
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import random
from functools import partial

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%% get influx data

index_client_df = idb.DataFrameClient(database='index')
index_client = idb.InfluxDBClient(database='index')
index_client.query('show measurements')
keys = index_client.query('show field keys')

#%%
symbol = 'DAX'
tz = pytz.timezone('Europe/Berlin')

def create_interactive_plot(ax, df, day):
  fplt.candlestick_ochl(df[['o', 'c', 'h', 'l']], ax=ax)
  hover_label = fplt.add_legend(f'<span style="font-size:15px;color:darkgreen;background-color:#fff">{day}</span>', ax=ax, )

  #######################################################
  ## update crosshair and legend when moving the mouse ##
  def update_legend_text(df_ul, hover_label_ul, x, y):
    # print(x)
    row = df_ul.loc[pd.to_datetime(x, unit='ns', utc=True)]
    # print(row)
    # format html with the candle and set legend
    fmt = '<span style="font-size:15px;color:#%s;background-color:#fff">%%.2f</span>' % ('0d0' if (row.o<row.c).all() else 'd00')
    rawtxt = '<span style="font-size:14px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
    values = [row.o, row.c, row.h, row.l]
    hover_label_ul.setText(rawtxt % tuple([symbol, day.upper()] + values))

  def update_crosshair_text(df_ch, x, y, xtext, ytext):
    ytext = '%s (Close%+.2f)' % (ytext, (y - df_ch.iloc[x].c))
    return xtext, ytext

  fplt.set_mouse_callback(partial(update_legend_text, df, hover_label), ax=ax, when='hover')
  fplt.add_crosshair_info(partial(update_crosshair_text, df), ax=ax)

def get_candles_range(start, end, symbol, group_by_time=None):
  base_query = f'select first(o) as o, last(c) as c, max(h) as h, min(l) as l from {symbol} where time >= \'{start.isoformat()}\' and time < \'{end.isoformat()}\''
  if group_by_time is None:
    return base_query
  return base_query + f' group by time({group_by_time})'

#%%
first_day = dateutil.parser.parse('2023-01-01T00:00:00').replace(tzinfo=tz)
last_day = dateutil.parser.parse('2025-02-07T00:00:00').replace(tzinfo=tz)

df_day = index_client_df.query(get_candles_range(first_day, last_day, symbol, '1d'))
df_day_clean = df_day[symbol].tz_convert(tz).dropna()
#%%
axs = fplt.create_plot(symbol, rows=3)
create_interactive_plot(axs[0], df_day_clean, '1d')
fplt.plot(df_day_clean.h-df_day_clean.l, legend='diff', ax=axs[1])
fplt.plot(df_day_clean.o-df_day_clean.c, legend='oc', ax=axs[2])
fplt.show()

#%%

# df_diff = df_day_clean.o-df_day_clean.c
df_diff = ((df_day_clean.o-df_day_clean.c) / df_day_clean.o )*100
df_diff.plot.hist(bins=100, alpha=0.7, color='blue', title='Histogram of Values')
fplt.show()

#%%

df_diff[(df_diff > -1.3) & (df_diff < 1.3) ].count()

22435 * 1.013
#%%
# eval_range_start = datetime.timedelta(hours=0, minutes=0)
# eval_range_end = datetime.timedelta(hours=23, minutes=0)
time_ranges = ['2m', '4m', '8m', '16m', '32m', '64m']

first_date_str = index_client.query(f'select first(o) from {symbol}').raw['series'][0]['values'][0][0]
last_date_str = index_client.query(f'select last(o) from {symbol}').raw['series'][0]['values'][0][0]

first_date = dateutil.parser.parse(first_date_str)
last_date = dateutil.parser.parse(last_date_str)
num_days = (last_date - first_date).days

# last_day = dateutil.parser.parse('2025-02-07T00:00:00').replace(tzinfo=tz)
# while first_day < last_day:
while True:
  selected_day_dist = random.randint(0, num_days)
  ref_range_start = first_date + datetime.timedelta(days=selected_day_dist)
  ref_range_end = ref_range_start + datetime.timedelta(days=1)
  dfs_day = []
  for time_range in time_ranges:
    # day_end = selected_day + datetime.timedelta(days=1)
    # print(
    #   f"getting data for {ref_range_start.isoformat()} - {ref_range_end.isoformat()}"
    # )
    df_day = index_client_df.query(get_candles_range(ref_range_start, ref_range_end, symbol, time_range))

    if df_day:
      dfs_day.append(df_day[symbol].tz_convert(tz).dropna())
    # selected_day = day_end

  if not dfs_day:
    print(f"no data for {ref_range_start.isoformat()} - {ref_range_end.isoformat()}")
    continue

  axs = fplt.create_plot(symbol, rows=len(dfs_day))
  for (ax, df) in zip(axs, dfs_day):
    create_interactive_plot(ax, df, ref_range_start.strftime('%Y-%m-%d'))

  fplt.show()

#%%
df_change = dfs_ref_range[0]

df_change['hh'] = df_change.h.shift(-1) - df_change.h
df_change['ll'] = df_change.l.shift(-1) - df_change.l

# df_change = pd.concat([df_change['hh'], df_change['ll']], axis=1)
df_change['pos'] = df_change.apply(lambda x: -1 if x.hh < 0 and x.ll < 0 else 1 if x.hh > 0 and x.ll > 0 else 0, axis=1)



# fplt.plot(df_change.hh, legend='HH', ax=ax[1])
# fplt.plot(df_change.ll, legend='LL', ax=ax[2])
# fplt.plot(df_change.pos, legend='pos', ax=ax[3])


# dax_df = df_change
symbol = 'DAX'
interval = '10m'
axs = fplt.create_plot('DAX', rows=len(dfs_day))
for (ax, df) in zip(axs, dfs_day):
  create_interactive_plot(ax, df)
fplt.show()
