#%%
import pickle

import numpy as np

import pandas as pd

import matplotlib as mpl

import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


##%% Get liquid underlyings
with open('finance/_data/liquid_stocks.pkl', 'rb') as f: liquid_symbols = pickle.load(f)

spy_data = utils.swing_trading_data.SwingTradingData('SPY')
df_spy_day = spy_data.df_day

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

  atr2xs = df_day[df_day.pct.abs() > 2 * df_day.atrp20]
  atr2x_data = []
  for atr2x in atr2xs.itertuples():
    idx = df_day.index.get_loc(atr2x[0])
    df_atr2x = df_day.iloc[idx, :]
    df_tracking_data = df_day.iloc[idx-1:idx+21, :][['c', 'v', 'atrp9', 'atrp14', 'atrp20', 'ac_lag_1', 'ac_lag_5', 'ac_lag_21', 'pct', 'rvol', 'iv', 'hv30']]
    df_tracking_data['cpct'] = (df_tracking_data['c'] - df_tracking_data['c'].iloc[0])/ df_tracking_data['c'].iloc[0] * 100

    idx_spy = df_spy_day.index.get_loc(atr2x[0])
    spy_ref_value = df_spy_day['c'].iloc[idx_spy-1]
    df_tracking_data[f'spy'] = (df_spy_day['c'].iloc[idx_spy-1:idx_spy+21] - spy_ref_value) / spy_ref_value * 100
    postEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx-1]), 'when']
    preEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx]), 'when']
    hasEarnings = postEarnings.iat[0] == 'post' if not postEarnings.empty else preEarnings.iat[0] == 'pre' if not preEarnings.empty else False
    atr2x_eval = {'symbol': ticker, 'earnings': hasEarnings, 'date': df_tracking_data.index[0], 'gappct': df_atr2x.gappct, 'c': df_atr2x.c,
                'is_etf': swing_data.symbol_info.is_etf, 'atrp20': df_atr2x.atrp20}

    if not swing_data.symbol_info.is_etf and swing_data.market_cap is not None:
      nearest_mc = swing_data.market_cap.iloc[swing_data.market_cap.index.get_indexer([atr2x[0]], method='nearest')[0]]
      atr2x_eval['market_cap'] = nearest_mc['market_cap']
      atr2x_eval['market_cap_date'] = nearest_mc.name

    # Elegant flattening:
    # 1. Set index to the desired suffixes (-1, 0, 1...)
    df_tracking_data.index = np.arange(len(df_tracking_data)) - 1

    # 2. Unstack turns it into a Series with a MultiIndex (Column, Suffix)
    # 3. Use a dictionary comprehension to join them into 'c-1', 'rvol0', etc.
    flat_metrics = {f"{col}{idx}": val for (col, idx), val in df_tracking_data.unstack().items()}
    atr2x_eval.update(flat_metrics)
    atr2x_data.append(atr2x_eval)
  df_gap_stats = pd.DataFrame(atr2x_data)

  #%%
  df_gap_stats.to_pickle(f'finance/_data/atr2x/{ticker}.pkl')

#%% Combine all gaps
# ticker = liquid_symbols[0]
start_at = liquid_symbols.index('AGG')
# start_at = 0
atr2x_data = []
for ticker in liquid_symbols[start_at:]:
  #%%
  print(f'Processing {ticker}...')
  df_ticker = pd.read_pickle(f'finance/_data/atr2x/{ticker}.pkl')
  if df_ticker.empty: continue
  atr2x_data.append(df_ticker)

#%%
df_atr2xs = pd.concat(atr2x_data)
df_atr2xs.to_pickle(f'finance/_data/all_atr2x.pkl')

