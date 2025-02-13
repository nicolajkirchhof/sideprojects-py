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
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%


#%% get influx data

index_client_df = idb.DataFrameClient(database='index')
index_client = idb.InfluxDBClient(database='index')
index_client.query('show measurements')
index_client.query('show field keys')
#%%
symbol = 'DAX'
def get_candles_range(start, end, group_by_time=None):
  base_query = f'select first(o) as o, last(c) as c, max(h) as h, min(l) as l from DAX where time >= \'{start.isoformat()}\' and time < \'{end.isoformat()}\''
  if group_by_time is None:
    return base_query
  return base_query + f' group by time({group_by_time})'

tz = pytz.timezone('Europe/Berlin')

dfs_ref_range = []
dfs_closing = []
first_day = tz.localize(dateutil.parser.parse('2015-01-01T00:00:00'))
last_day = tz.localize(dateutil.parser.parse('2015-02-07T00:00:00'))
while first_day < last_day:
  # day_end = dateutil.parser.parse('2025-02-01T00:00:00').replace(tzinfo=pytz.timezone('Europe/Berlin'))
  day_end = first_day + datetime.timedelta(days=1)
  # get the following data for daily assignment
  # overnight range 0-7:00
  # onight_stop = day_start + datetime.timedelta(hours=7)
  # df_onight_range = index_client_df.query(get_candles_range(day_start, onight_stop))
  # 10min candles from 9:00 - 10:00
  ref_range_start = first_day + datetime.timedelta(hours=9)
  ref_range_end = first_day + datetime.timedelta(hours=10)
  df_ref_range = index_client_df.query(get_candles_range(ref_range_start, ref_range_end, '10m'))

  # End of day value at 16:00
  before_closing = first_day + datetime.timedelta(hours=15)
  closing = first_day + datetime.timedelta(hours=16)
  df_closing = index_client_df.query(get_candles_range(before_closing, closing))
  if df_ref_range:
    dfs_ref_range.append(df_ref_range[symbol].tz_convert(tz))
  if df_closing:
    dfs_closing.append(df_closing[symbol].tz_convert(tz))
  first_day = day_end

#%%
dfs_diff = []
df_result = []
for df_in, df_out in zip(dfs_ref_range, dfs_closing):
  df_input = (df_in.c - df_in.o) / (df_in.h - df_in.l)
  dfs_diff.append(df_input.reset_index(drop=True))
  df_result.append(percentage_change(df_out.c.iat[0],df_in.iloc[-1].c))

#%%
dfs_diff_df = pd.concat(dfs_diff, axis=1).T

#%%

# 2. Split Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(dfs_diff_df, df_result, test_size=0.25, random_state=42)

# Train Logistic Regression Model
# model = LogisticRegression()
model = LinearRegression()
model.fit(X_train, Y_train)

# Evaluate
Y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.2f}")

#%%
# 2. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(dfs_diff_df, df_result, test_size=0.25, random_state=42)

# 3. Train the Linear Regression Model
model = LinearRegression()  # Initialize the model
model.fit(X_train, y_train)  # Train the model on the training data

# 4. Make Predictions
y_pred = model.predict(X_test)  # Predict on the test data

# 5. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
r2 = r2_score(y_test, y_pred)  # Calculate R² Score

print(f"Model Coefficients (slope): {model.coef_[0][0]:.2f}")
print(f"Model Intercept: {model.intercept_[0]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


# Get DAX Data for one day to find alignment
# day_start = dateutil.parser.parse('2024-10-29T00:00:00+01:00')
# day_end = day_start + datetime.timedelta(days=1)


# data = index_client_df.query(query)
# dax_df = data['DAX']
# dax_df = dax_df.tz_convert(tz)
#%%
symbol = 'DAX'
interval = '10m'
ax = fplt.create_plot('DAX', rows=1)
fplt.candlestick_ochl(dax_df[['o', 'c', 'h', 'l']])
fplt.add_legend('', ax=ax)

hover_label = fplt.add_legend('', ax=ax)

#######################################################
## update crosshair and legend when moving the mouse ##

def update_legend_text(x, y):
  # print(x)
  row = dax_df.loc[pd.to_datetime(x, unit='ns', utc=True)]
  # print(row)
  # format html with the candle and set legend
  fmt = '<span style="font-size:15px;color:#%s;background-color:#fff">%%.2f</span>' % ('0d0' if (row.o<row.c).all() else 'd00')
  rawtxt = '<span style="font-size:14px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
  values = [row.o, row.c, row.h, row.l]
  hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))

def update_crosshair_text(x, y, xtext, ytext):
  ytext = '%s (Close%+.2f)' % (ytext, (y - dax_df.iloc[x].c))
  return xtext, ytext

fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
fplt.add_crosshair_info(update_crosshair_text, ax=ax)
fplt.show()
