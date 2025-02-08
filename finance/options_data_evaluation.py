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
import finplot as fplt

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%
# Get DAX Data for one day to find alignment
day_start = dateutil.parser.parse('2024-10-29T00:00:00+01:00')
day_end = day_start + datetime.timedelta(days=1)

#%% get influx data

index_client_df = idb.DataFrameClient(database='index')
index_client = idb.InfluxDBClient(database='index')
index_client.query('show measurements')
index_client.query('show field keys')
#%%
query = f'select  first(o) as o, last(c) as c, max(h) as h, min(l) as l from DAX where time >= \'{day_start.isoformat()}\' and time < \'{day_end.isoformat()}\' group by time(10m)'
data = index_client_df.query(query)
dax_df = data['DAX']

#%%
dax_df.plot()
plt.show()

#%%
def update_legend_text(x, y):
  row = df.loc[pd.to_datetime(x, unit='ns')]
  # format html with the candle and set legend
  fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open<row.Close).all() else 'a00')
  rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
  values = [row.Open, row.Close, row.High, row.Low]
  hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))

fplt.candlestick_ochl(dax_df[['o', 'c', 'h', 'l']])
fplt.show()
