# %%
import os
import pickle

import numpy as np

import pandas as pd

import matplotlib as mpl

import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%% Get liquid underlyings
# df_liquid_symbols = pd.read_csv('finance/_data/stocks-screener-eval-stocks-etfs-01-07-2026.csv', skipfooter=1, engine='python', keep_default_na=False, na_values=['N/A'])
liquid_symbols = pickle.load(open('finance/_data/liquid_symbols.pkl', 'rb'))

spy_data = utils.swing_trading_data.SwingTradingData('SPY')
df_spy_day = spy_data.df_day

# %%
# ticker = df_liquid_symbols[0]
start_at = 0
start_at = liquid_symbols.index('PCOR')
# TO DO AGAIN RGEN
# start_at = df_liquid_symbols[df_liquid_symbols.Symbol == 'OUT'].index[0] + 1
end_at = len(liquid_symbols)
##%%
# for ticker in df_liquid_symbols[start_at:end_at]:
for ticker in liquid_symbols[start_at:end_at]:
  if os.path.exists(f'finance/_data/ibkr/stk/{ticker}.pkl'): continue
  print(f'Processing {ticker}...')
  # %%
  df_earnings = pd.read_csv(f'finance/_data/earnings_cleaned/{ticker}.csv')
  df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')

  swing_data = utils.swing_trading_data.SwingTradingData(ticker)
  if swing_data.empty or swing_data.info is None: continue
  df_day = swing_data.df_day
  # %% Dataset of gaps: gap%, after 3 days, after 6 days, after 9 days, after 14 days
  PREV_DAYS = 5

  atr2xs = df_day[df_day.pct.abs() > 2 * df_day.atrp20]
  atr2x_data = []
  for atr2x in atr2xs.itertuples():
    idx = df_day.index.get_loc(atr2x[0])
    if idx < PREV_DAYS: continue
    df_atr2x = df_day.iloc[idx, :]
    df_tracking_data = df_day.iloc[idx - PREV_DAYS:idx + 21, :][
      ['c', 'v', 'atrp9', 'atrp14', 'atrp20', 'ac100_lag_1', 'ac100_lag_5', 'ac100_lag_20', 'ac_comp', 'ac_mom', 'ac_mr',
       'ac_inst', 'pct', 'rvol20', 'rvol50', 'iv', 'hv9', 'hv14', 'hv20', 'hv60', 'ema20_dist', 'ema10_dist', 'ema10_slope', 'ema20_slope']]
    df_tracking_data['cpct'] = (df_tracking_data['c'] - df_tracking_data['c'].iloc[PREV_DAYS-1]) / df_tracking_data['c'].iloc[PREV_DAYS-1] * 100

    if not any(df_spy_day.index == atr2x[0]):
      print(
        f'No SPY data for {ticker} at {atr2x[0]} (idx {idx}). Skipping gap evaluation.'
      )
      continue
    idx_spy = df_spy_day.index.get_loc(atr2x[0])
    spy_ref_value = df_spy_day['c'].iloc[idx_spy - PREV_DAYS]
    df_tracking_data[f'spy'] = (df_spy_day['c'].iloc[idx_spy - PREV_DAYS:idx_spy + 21] - spy_ref_value) / spy_ref_value * 100
    postEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx - 1]), 'when']
    preEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx]), 'when']
    hasEarnings = postEarnings.iat[0] == 'post' if not postEarnings.empty else preEarnings.iat[
                                                                                 0] == 'pre' if not preEarnings.empty else False
    atr2x_eval = {'symbol': ticker, 'earnings': hasEarnings, 'date': df_tracking_data.index[0],
                  'gappct': df_atr2x.gappct, 'c': df_atr2x.c,
                  'is_etf': swing_data.info.is_etf, 'atrp20': df_atr2x.atrp20}

    if not swing_data.info.is_etf and swing_data.market_cap is not None and swing_data.market_cap.empty is False:
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

  # %%
  df_gap_stats.to_pickle(f'finance/_data/atr2x/{ticker}.pkl')

# %% Combine all gaps
# ticker = liquid_symbols[0]
# start_at = liquid_symbols.index('AGG')
start_at = 0
atr2x_data = []
for ticker in liquid_symbols[start_at:]:
  # %%
  print(f'Processing {ticker}...')
  df_ticker = pd.read_pickle(f'finance/_data/atr2x/{ticker}.pkl')
  if df_ticker.empty: continue
  atr2x_data.append(df_ticker)

## %%
df_atr2xs = pd.concat(atr2x_data)
df_atr2xs.to_pickle(f'finance/_data/all_atr2x.pkl')
