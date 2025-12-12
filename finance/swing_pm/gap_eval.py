#%%
import glob
import pickle
from datetime import datetime, timedelta
from glob import glob
from zoneinfo import ZoneInfo

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
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mssql.information_schema import columns

import finance.utils as utils

import yfinance as yf
import requests

from finance.swing_pm.earnings_dates import EarningsDates

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


##%%

ticker = 'MSFT'
# MySQL connection setup (localhost:3306)
# Note: This requires a driver like 'pymysql'. Install it via: pip install pymysql
# Format: mysql+driver://user:password@host:port/database
db_connection_str = 'mysql+pymysql://root:@localhost:3306/stocks'
db_connection = create_engine(db_connection_str)
# for ticker in utils.underlyings.us_stock_symbols:
##%% Get liquid underlyings
with open('finance/_data/liquid_stocks.pkl', 'rb') as f: liquid_symbols = pickle.load(f)

ticker = liquid_symbols[0]

#%%
df_earnings = pd.read_csv(f'finance/_data/earnings_cleaned/{ticker}.csv')
df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')

#%%
query = """select * from ohlcv where act_symbol = :ticker"""
stmt = text(query)
# Example usage with pandas:
df_stk = pd.read_sql(stmt, db_connection, params={'ticker': ticker})

df_stk = df_stk.rename(columns={'act_symbol':'symbol', 'open':'o', 'close':'c', 'high':'h', 'low':'l', 'volume':'v'})
df_stk['date'] = pd.to_datetime(df_stk.date)

#%%
df_stk['gap'] = df_stk.o - df_stk.shift().c
df_stk['gappct'] = utils.pct.percentage_change_array(df_stk.shift().c, df_stk.o)
df_stk['pct'] = utils.pct.percentage_change_array(df_stk.shift().c, df_stk.c)

##%% Dataset of gaps: gap%, after 3 days, after 6 days, after 9 days, after 14 days
gapdb = {'date':None, '': None, 'a3days': None, 'a6days': None, 'a9days': None, 'a14days': None, 'earnings': None}
gap_ups = df_stk[df_stk.gappct > 3]
gap_downs = df_stk[df_stk.gappct < -3]

#%%
def plot_gap_data_arrays(gap_data, alpha=0.2, lw=1.0, show_mean=True):
  """
  gap_data: iterable of 1D arrays (each array may have different length)
  Plots each array as y over x = 0..len(y)-1
  """
  fig, ax = plt.subplots(figsize=(12, 6))

  ys = []
  for y in gap_data:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
      continue
    x = np.arange(y.size)
    ax.plot(x, y, color="tab:blue", alpha=alpha, linewidth=lw)
    ys.append(y)

  if show_mean and ys:
    max_len = max(len(y) for y in ys)
    mat = np.full((len(ys), max_len), np.nan, dtype=float)
    for i, y in enumerate(ys):
      mat[i, :len(y)] = y
    mean_y = np.nanmean(mat, axis=0)
    ax.plot(np.arange(max_len), mean_y, color="black", linewidth=2.5, label=f"Mean (n={len(ys)})")
    ax.legend()

  ax.set_xlabel("Index (0..len(y)-1)")
  ax.set_ylabel("y")
  ax.grid(True, alpha=0.25)
  plt.tight_layout()
  return fig, ax

#%%
def plot_gaps(gaps):
  gap_data = {'earnings': None, 'data': None}
  for gap in gaps.itertuples():
    idx = gap[0]
    closes = df_stk.iloc[idx-1:idx+14, :]['c'].to_numpy()
    closes_norm = (closes-closes[0])/closes[0] * 100
    postEarnings = df_earnings.loc[df_earnings.date.eq(df_stk.iloc[idx-1].date), 'when']
    preEarnings = df_earnings.loc[df_earnings.date.eq(df_stk.iloc[idx].date), 'when']
    hasEarnings = postEarnings.iat[0] == 'post' if not postEarnings.empty else preEarnings.iat[0] == 'pre' if not preEarnings.empty else False
    gap_data.append({'earnings': hasEarnings, 'data': closes_norm})

  plot_gap_data_arrays(gap_data)
  plt.show()

#%%
plot_gaps(gap_ups)
plot_gaps(gap_downs)
#%%
gap = next(gaps.itertuples())
for gap in gaps.itertuples():
  idx = gap[0]
  #%%
  aXdays = []
  for i in [1, 3, 5, 7, 10, 14]:
    data = df_stk.iloc[idx + i, :]
    data.rename(columns={'date': f'date{i}', 'o': f'o{i}', 'h': f'h{i}', 'c':f'c{i}', 'l':f'l{i}', 'v': f'v{i}', 'gap'})
    aXdays.append(data)

