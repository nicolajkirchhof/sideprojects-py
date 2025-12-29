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

from finance.ibkr.options_gex import last_expiration
from finance.swing_pm.earnings_dates import EarningsDates

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
def gap_statistics(gaps):
  gap_data = []
  for gap in gaps.itertuples():
    idx = gap[0]
    closes = df_stk.iloc[idx-1:idx+21, :]['c'].to_numpy()
    closes_norm = (closes-closes[0])/closes[0] * 100
    postEarnings = df_earnings.loc[df_earnings.date.eq(df_stk.iloc[idx-1].date), 'when']
    preEarnings = df_earnings.loc[df_earnings.date.eq(df_stk.iloc[idx].date), 'when']
    hasEarnings = postEarnings.iat[0] == 'post' if not postEarnings.empty else preEarnings.iat[0] == 'pre' if not preEarnings.empty else False
    gap_eval = {'symbol': getattr(gap, 'symbol'), 'earnings': hasEarnings, 'date': getattr(gap, 'date'), 'gappct':getattr(gap, 'gappct') }
    for i, y in enumerate(closes_norm):
      gap_eval[f't{i-1}']=y
    gap_data.append(gap_eval)
  return pd.DataFrame(gap_data)


# MySQL connection setup (localhost:3306)
# Note: This requires a driver like 'pymysql'. Install it via: pip install pymysql
# Format: mysql+driver://user:password@host:port/database
db_connection_str = 'mysql+pymysql://root:@localhost:3306/stocks'
db_connection = create_engine(db_connection_str)
# for ticker in utils.underlyings.us_stock_symbols:
##%% Get liquid underlyings
with open('finance/_data/liquid_stocks.pkl', 'rb') as f: liquid_symbols = pickle.load(f)

#%%
ticker = liquid_symbols[0]
for ticker in liquid_symbols:
  print(f'Processing {ticker}...')
  #%%
  df_earnings = pd.read_csv(f'finance/_data/earnings_cleaned/{ticker}.csv')
  df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')

  swing_data = utils.swing_trading_data.SwingTradingData(ticker)
  df_day = swing_data.df_day

  ##%% Dataset of gaps: gap%, after 3 days, after 6 days, after 9 days, after 14 days
  gaps = df_day[df_day.gappct.abs() > 2]
  gap_data = []
  for gap in gaps.itertuples():
    idx = df_day.index.get_loc(gap[0])
    df_gap = df_day.iloc[idx, :]
    df_tracking_data = df_day.iloc[idx-1:idx+21, :][['c', 'v' 'rvol', 'iv', 'hv30']]
    df_tracking_data['cpct'] = df_tracking_data['c'] / df_tracking_data['c'].iloc[0] * 100
    postEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx-1]), 'when']
    preEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx]), 'when']
    hasEarnings = postEarnings.iat[0] == 'post' if not postEarnings.empty else preEarnings.iat[0] == 'pre' if not preEarnings.empty else False
    latestEarnings = df_earnings.loc[df_earnings.date >= df_day.index[idx]].iloc[0]
    gap_eval = {'symbol': ticker, 'earnings': hasEarnings, 'date': df_tracking_data.index[0], 'gappct': df_gap.gappct, 'c': df_gap.c,
                'is_etf': swing_data.symbol_info.is_etf}
    # Elegant flattening:
    # 1. Set index to the desired suffixes (-1, 0, 1...)
    df_tracking_data.index = np.arange(len(df_tracking_data)) - 1

    # 2. Unstack turns it into a Series with a MultiIndex (Column, Suffix)
    # 3. Use a dictionary comprehension to join them into 'c-1', 'rvol0', etc.
    flat_metrics = {f"{col}{idx}": val for (col, idx), val in df_tracking_data.unstack().items()}
    gap_eval.update(flat_metrics)
    gap_data.append(gap_eval)


  #%%
  gap_stats_df = gap_statistics(gaps)
  gap_stats_df.to_pickle(f'finance/_data/gaps/{ticker}.pkl')

#%%
dfs_gaps = []
for ticker in liquid_symbols:
  df_ticker_gaps = pd.read_pickle(f'finance/_data/gaps/{ticker}.pkl')
  df_ticker_data = utils.dolt_data.daily_time_range(ticker, df_ticker_gaps.date.min(), df_ticker_gaps.date.max())

df_gaps = pd.concat(dfs_gaps)

df_gaps_clean = df_gaps.dropna(how='any')

#%% Add symbol information
stmt = text("select * from symbol where act_symbol in :tickers").bindparams(bindparam("tickers", expanding=True))
df_symbols = pd.read_sql(stmt, db_connection, params={'tickers': liquid_symbols})
df_symbols = df_symbols.rename(columns={'act_symbol':'symbol', 'security_name': 'name'})

#%%
df_gaps_sym = pd.merge(df_gaps_clean, df_symbols[['symbol','name', 'is_etf']], on=['symbol'], how='inner')
df_gaps_sym.to_pickle(f'finance/_data/gaps_sym.pkl')

