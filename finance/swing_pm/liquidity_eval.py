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

import finance.utils as utils

import yfinance as yf
import requests

from finance.swing_pm.earnings_dates import EarningsDates

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%

ticker = 'MSFT'
# MySQL connection setup (localhost:3306)
# Note: This requires a driver like 'pymysql'. Install it via: pip install pymysql
# Format: mysql+driver://user:password@host:port/database
db_connection_str = 'mysql+pymysql://root:@localhost:3306/stocks'
db_connection = create_engine(db_connection_str)
# for ticker in utils.underlyings.us_stock_symbols:

#%% SQL-side Quantile Calculation for ALL tickers
# Using Window Functions partitioned by symbol to get stats for everyone at once
all_stats_query = text("""
                       WITH Ordered AS (
                           SELECT
                               act_symbol,
                               volume,
                               ROW_NUMBER() OVER (PARTITION BY act_symbol ORDER BY volume) as rn,
                               COUNT(*) OVER (PARTITION BY act_symbol) as total
                           FROM ohlcv
                           WHERE date >= '2020-01-01'
                       )
                       SELECT
                           act_symbol as symbol,
                           MAX(CASE WHEN rn = 1 THEN volume END) as min_val,
                           MAX(CASE WHEN rn = GREATEST(1, FLOOR(total * 0.25)) THEN volume END) as q1,
                           MAX(CASE WHEN rn = GREATEST(1, FLOOR(total * 0.50)) THEN volume END) as median,
                           MAX(CASE WHEN rn = GREATEST(1, FLOOR(total * 0.75)) THEN volume END) as q3,
                           MAX(CASE WHEN rn = total THEN volume END) as max_val
                       FROM Ordered
                       GROUP BY act_symbol
                       """)

df_all_stats = pd.read_sql(all_stats_query, db_connection)
# Set symbol as index for easier lookups
df_all_stats.set_index('symbol', inplace=True)
print(df_all_stats.head())
df_all_stats.to_pickle(f'finance/_data/stocks_liquidity_stats.pkl')
#%%
liquid_names = df_all_stats[df_all_stats['median'] > 1000000].index.tolist()
with open('finance/_data/liquid_stocks.pkl', 'wb') as f: pickle.dump(liquid_names, f)

