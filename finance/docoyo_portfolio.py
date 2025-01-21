# %%
import os

import yfinance as yf
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import finplot as fplt
import mplfinance as mpf

from finance.yf_data_aquisition import df_etfs

mpl.use('TkAgg')
mpl.use('QtAgg')

# %% ETFS
df_etfs = pd.read_csv('finance/etf_symbols.csv')

# %%
etf_data = {}
for etf_symbol in df_etfs['symbol']:
  filename = f'finance/y_finance_data/{etf_symbol}.csv'
  if os.path.isfile(filename):
    etf_data[etf_symbol] = pd.read_csv(filename)
    continue
  etf_data[etf_symbol] = yf.Ticker(etf_symbol).history(start="2015-01-01", interval="1d")
  pd.DataFrame(etf_data[etf_symbol]).to_csv(filename)

# %%

portfolios = [k for k in df_etfs.keys() if 'p_' in k]
df_portfolios = {}
for p in portfolios:
  df_portfolios[p] = None
  for index, row in df_etfs.iterrows():
    print(row['symbol'], p)
    df = row[p] * etf_data[row['symbol']][['Open', 'High', 'Low', 'Close']] / etf_data[row['symbol']]['Open'][0]
    if df_portfolios[p] is not None:
        df_portfolios[p] = df_portfolios[p] + df
    else:
      df_portfolios[p] = df

# %%

fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
fplt.show()

# %%
fig, ax = mpf.plot(df, returnfig=True, volume=True, type='candle', tight_layout=True)
