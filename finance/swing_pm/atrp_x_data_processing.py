# %%
import os
import pickle
from datetime import timedelta

import numpy as np

import pandas as pd

import matplotlib as mpl

import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%% Get liquid underlyings
liquid_symbols = pickle.load(open('finance/_data/liquid_symbols.pkl', 'rb'))

spy_data = utils.swing_trading_data.SwingTradingData('SPY', offline=True)

name = 'atr_x'
os.makedirs(f'finance/_data/{name}', exist_ok=True)
#%%
def fn_flt(df):
  return 1.5*df['atrp20'] < df.pct.abs()

utils.momentum.evaluate_liquid_symbols(name, fn_flt, spy_data, liquid_symbols, start_at_name='CIDM')

#%%
def calculate_performance(df_ticker, df_day, length_days):
  df_c = df_day.iloc[df_day.index.get_indexer(df_ticker.date - timedelta(days=length_days), method='ffill')]['c'].copy()
  df_diff = df_ticker.date - df_c.index
  df_c[(df_diff.abs() > timedelta(days=length_days + 5)).values] = np.nan

  return df_c
# %% Combine all gaps
# ticker = liquid_symbols[0]
# start_at = liquid_symbols.index('AGG')
start_at = 0
# ticker = liquid_symbols[42]
# atr2x_data = []
for ticker in liquid_symbols[start_at:]:
  # %%
  print(f'Processing {ticker}...')
  filename = f'finance/_data/{name}/{ticker}.pkl'
  if not os.path.exists(filename): continue
  df_ticker = pd.read_pickle(filename)
  if df_ticker.empty: continue
  swing_data = utils.swing_trading_data.SwingTradingData(ticker, offline=True, metainfo=False)
  df_day = swing_data.df_day
  #%%
  for tf_name, tf_days in [('1M', 30), ('3M', 90), ('6M', 180)]:
    df = calculate_performance(df_ticker, df_day, tf_days)
    df_ticker[tf_name] = df.values
    df_ticker[tf_name + '_chg'] = utils.pct.percentage_change_array(df_ticker['c-1'], df.values) # (df_ticker['c-1'] - df_1M.values) / df_1M.values * 100

  df_ticker.to_pickle(filename)
## %%
# df_atr2xs = pd.concat(atr2x_data)
# df_atr2xs.to_pickle(f'finance/_data/all_atr2x.pkl')
