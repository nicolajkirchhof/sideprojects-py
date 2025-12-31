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
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.mssql.information_schema import columns

import finance.utils as utils

import yfinance as yf
import requests

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


##%% Get liquid underlyings
with open('finance/_data/liquid_stocks.pkl', 'rb') as f: liquid_symbols = pickle.load(f)
#%%
# MySQL connection setup (localhost:3306)
# Note: This requires a driver like 'pymysql'. Install it via: pip install pymysql
# Format: mysql+driver://user:password@host:port/database
db_connection_str = 'mysql+pymysql://root:@localhost:3306/stocks'
db_connection = create_engine(db_connection_str)
# for ticker in utils.underlyings.us_stock_symbols:

#%%
ticker = liquid_symbols[0]
start_at = liquid_symbols.index('ARVL')
for ticker in liquid_symbols[start_at:]:
  print(f'Processing {ticker}...')
  #%%
  df_earnings = pd.read_csv(f'finance/_data/earnings_cleaned/{ticker}.csv')
  df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')

  swing_data = utils.swing_trading_data.SwingTradingData(ticker)
  df_day = swing_data.df_day

  #%% Dataset of gaps: gap%, after 3 days, after 6 days, after 9 days, after 14 days

  gaps = df_day[df_day.gappct.abs() > 2]
  gap_data = []
  for gap in gaps.itertuples():
    idx = df_day.index.get_loc(gap[0])
    df_gap = df_day.iloc[idx, :]
    df_tracking_data = df_day.iloc[idx-1:idx+21, :][['c', 'v', 'rvol', 'iv', 'hv30']]
    df_tracking_data['cpct'] = (df_tracking_data['c'] - df_tracking_data['c'].iloc[0])/ df_tracking_data['c'].iloc[0] * 100
    postEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx-1]), 'when']
    preEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx]), 'when']
    hasEarnings = postEarnings.iat[0] == 'post' if not postEarnings.empty else preEarnings.iat[0] == 'pre' if not preEarnings.empty else False
    gap_eval = {'symbol': ticker, 'earnings': hasEarnings, 'date': df_tracking_data.index[0], 'gappct': df_gap.gappct, 'c': df_gap.c,
                'is_etf': swing_data.symbol_info.is_etf}

    if not swing_data.symbol_info.is_etf and swing_data.market_cap is not None:
      nearest_mc = swing_data.market_cap.iloc[swing_data.market_cap.index.get_indexer([gap[0]], method='nearest')[0]]
      gap_eval['market_cap'] = nearest_mc['market_cap']
      gap_eval['market_cap_date'] = nearest_mc.name

  # Elegant flattening:
    # 1. Set index to the desired suffixes (-1, 0, 1...)
    df_tracking_data.index = np.arange(len(df_tracking_data)) - 1

    # 2. Unstack turns it into a Series with a MultiIndex (Column, Suffix)
    # 3. Use a dictionary comprehension to join them into 'c-1', 'rvol0', etc.
    flat_metrics = {f"{col}{idx}": val for (col, idx), val in df_tracking_data.unstack().items()}
    gap_eval.update(flat_metrics)
    gap_data.append(gap_eval)
  df_gap_stats = pd.DataFrame(gap_data)

  #%%
  df_gap_stats.to_pickle(f'finance/_data/gaps/{ticker}.pkl')


#%% Adding the movement of the SPY in the same range as information
# TODO integrate into the upper part when touching it again

spy_data = utils.swing_trading_data.SwingTradingData('SPY')
df_spy_day = spy_data.df_day
# ticker = liquid_symbols[0]
start_at = liquid_symbols.index('AGG')
# start_at = 0
for ticker in liquid_symbols[start_at:]:
  #%%
  print(f'Processing {ticker}...')
  df_ticker = pd.read_pickle(f'finance/_data/gaps/{ticker}.pkl')
  if df_ticker.empty: continue
  idx_spys = df_spy_day.index.get_indexer(df_ticker.date, method='nearest')
  for i, idx_spy in enumerate(idx_spys):
    #%%
    spy_ref_value = df_spy_day['c'].iloc[idx_spy-1]
    for j in range(-1, 21):
      idx_last = idx_spy+j
      if idx_spy+j < len(df_spy_day):
        df_ticker.loc[df_ticker.index[i], f'spy{j}'] = (df_spy_day['c'].iloc[idx_spy+j] - spy_ref_value) / spy_ref_value * 100
      else:
        df_ticker.loc[df_ticker.index[i], f'spy{j}'] = np.nan

  df_ticker.to_pickle(f'finance/_data/gaps/{ticker}.pkl')

#%% Combine all gaps
# ticker = liquid_symbols[0]
start_at = liquid_symbols.index('AGG')
# start_at = 0
gap_data = []
for ticker in liquid_symbols[start_at:]:
  #%%
  print(f'Processing {ticker}...')
  df_ticker = pd.read_pickle(f'finance/_data/gaps/{ticker}.pkl')
  if df_ticker.empty: continue
  gap_data.append(df_ticker)

#%%
df_gaps = pd.concat(gap_data)
df_gaps.to_pickle(f'finance/_data/all_gaps.pkl')

