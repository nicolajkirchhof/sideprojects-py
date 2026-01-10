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
liquid_symbols = pickle.load(open('finance/_data/liquid_stocks.pkl', 'rb'))

spy_data = utils.swing_trading_data.SwingTradingData('SPY')
df_spy_day = spy_data.df_day


# %%
ticker = liquid_symbols[42]
start_at = 0
# start_at = liquid_symbols.index('GNRC')
# TO DO AGAIN RGEN
# start_at = df_liquid_symbols[df_liquid_symbols.Symbol == 'OUT'].index[0] + 1
end_at = len(liquid_symbols)
#%%
# for ticker in df_liquid_symbols[start_at:end_at]:
for ticker in liquid_symbols[start_at:end_at]:
  if not os.path.exists(f'finance/_data/atr2x/{ticker}.pkl'): continue
  print(f'Processing {ticker}...')
  # %%
  df_earnings = pd.read_csv(f'finance/_data/earnings_cleaned/{ticker}.csv')
  df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')

  swing_data = utils.swing_trading_data.SwingTradingData(ticker, offline=True)
  df_day = swing_data.df_day
  df_week = swing_data.df_week
  # %% Dataset of gaps: gap%, after 3 days, after 6 days, after 9 days, after 14 days
  OFFSET_DAYS = 25

  i, earnings_event = next(df_earnings.iterrows())
  for i, earnings_event in df_earnings.iterrows():
    #%%
    idx = df_day.index.get_loc(earnings_event.date)
    reaction_idx = idx + 1 if earnings_event.when == 'post' else idx
    reaction_date = df_day.iloc[reaction_idx].name
    # if idx < PREV_DAYS: continue
    df_peds = df_day.iloc[reaction_idx, :]
    df_tracking_data = df_day.iloc[idx - OFFSET_DAYS:idx + OFFSET_DAYS, :][
      ['c', 'v', 'atrp9', 'atrp14', 'atrp20', 'ac100_lag_1', 'ac100_lag_5', 'ac100_lag_20', 'ac_comp', 'ac_mom', 'ac_mr',
       'ac_inst', 'pct', 'rvol20', 'rvol50', 'iv', 'hv9', 'hv14', 'hv30', 'hv60', 'ema20_dist', 'ema10_dist', 'ema10_slope', 'ema20_slope']]
    df_tracking_data['cpct'] = (df_tracking_data['c'] - df_tracking_data['c'].iloc[OFFSET_DAYS-1]) / df_tracking_data['c'].iloc[OFFSET_DAYS-1] * 100

    if not any(df_spy_day.index == reaction_date):
      print(
        f'No SPY data for {ticker} at {reaction_date} (idx {idx}). Skipping gap evaluation.'
      )
      raise()
      # continue
    idx_spy = df_spy_day.index.get_loc(reaction_date)
    spy_ref_value = df_spy_day['c'].iloc[idx_spy - OFFSET_DAYS]
    df_tracking_data[f'spy'] = (df_spy_day['c'].iloc[idx_spy - OFFSET_DAYS:idx_spy + OFFSET_DAYS] - spy_ref_value) / spy_ref_value * 100
    #%%
    peads_eval = {'gappct': df_peds.gappct}

    if swing_data.market_cap is not None and swing_data.market_cap.empty is False:
      nearest_mc = swing_data.market_cap.iloc[swing_data.market_cap.index.get_indexer([reaction_date], method='nearest')[0]]
      peads_eval['market_cap'] = nearest_mc['market_cap']
      peads_eval['market_cap_date'] = nearest_mc.name

    # Elegant flattening:
    # 1. Set index to the desired suffixes (-1, 0, 1...)
    df_tracking_data.index = np.arange(len(df_tracking_data)) - OFFSET_DAYS

    # 2. Unstack turns it into a Series with a MultiIndex (Column, Suffix)
    # 3. Use a dictionary comprehension to join them into 'c-1', 'rvol0', etc.
    flat_metrics = {f"{col}{idx}": val for (col, idx), val in df_tracking_data.unstack().items()}
    peads_eval.update(flat_metrics)
    peads_data.append(peads_eval)
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
