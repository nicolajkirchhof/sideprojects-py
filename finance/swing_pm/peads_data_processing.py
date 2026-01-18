# %%
import os
import pickle
from datetime import timedelta

import numpy as np

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%% Get liquid underlyings
liquid_stocks = pickle.load(open('finance/_data/liquid_stocks.pkl', 'rb'))

spy_data = utils.swing_trading_data.SwingTradingData('SPY', offline=True)
df_spy_day = spy_data.df_day
df_spy_week = spy_data.df_week


# %%
# ticker = liquid_stocks[42]
ticker = 'OXLC'
start_at = 0
start_at = liquid_stocks.index(ticker)
# TO DO AGAIN RGEN
# start_at = df_liquid_symbols[df_liquid_symbols.Symbol == 'OUT'].index[0] + 1
end_at = len(liquid_stocks)
#%%
for ticker in liquid_stocks[start_at:end_at]:
  # if os.path.exists(f'finance/_data/peads/{ticker}.pkl'): continue
  print(f'Processing {ticker}...')

  os.makedirs(f'finance/_data/peads/plots/{ticker}', exist_ok=True)
  # %%
  df_earnings = pd.read_csv(f'finance/_data/earnings_cleaned/{ticker}.csv')
  df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')

  swing_data = utils.swing_trading_data.SwingTradingData(ticker, offline=True)
  df_day = swing_data.df_day
  df_week = swing_data.df_week
  # %% Dataset of gaps: gap%, after 3 days, after 6 days, after 9 days, after 14 days
  OFFSET_DAYS = 25
  OFFSET_WEEKS = 8

  # i, earnings_event = next(df_earnings.iterrows())
  peads_data = []
  #%%
  for i, earnings_event in df_earnings.iterrows():
    if df_day.index.min() > earnings_event.date - timedelta(days=OFFSET_DAYS) or df_day.index.max() < earnings_event.date + timedelta(days=OFFSET_DAYS): continue

    #%%
    idx = df_day.index.get_indexer([earnings_event.date], method='nearest')[0]
    reaction_idx = idx + 1 if earnings_event.when == 'post' else idx
    reaction_date = df_day.iloc[reaction_idx].name
    # if idx < PREV_DAYS: continue
    df_peds = df_day.iloc[reaction_idx, :]
    df_tracking_data = df_day.iloc[reaction_idx - OFFSET_DAYS:reaction_idx + OFFSET_DAYS, :][utils.momentum.INDICATORS]
    df_plotting_data = df_day.iloc[reaction_idx - OFFSET_DAYS:reaction_idx + OFFSET_DAYS, :]
    if len(df_tracking_data) < 2 * OFFSET_DAYS or all(df_plotting_data.c.isna()):
      print(f'Tracking data too short for {ticker} at {earnings_event.date}. Skipping earnings evaluation.')
      # raise('tracking data too short')
      continue
    df_tracking_data['cpct'] = (df_tracking_data['c'] - df_tracking_data['c'].iloc[OFFSET_DAYS-1]) / df_tracking_data['c'].iloc[OFFSET_DAYS-1] * 100

    idx_week = df_week.index.get_indexer([earnings_event.date], method='nearest')[0]
    idx_week_offset = OFFSET_WEEKS if idx_week >= OFFSET_WEEKS else idx_week
    idx_start_week = idx_week - OFFSET_WEEKS if idx_week - OFFSET_WEEKS >= 0 else 0
    idx_end_week = idx_week + OFFSET_WEEKS if idx_week + OFFSET_WEEKS <= len(df_week) - 1 else len(df_week) - 1
    idx_end_week_offset = idx_end_week-idx_start_week-idx_week_offset
    df_plotting_data_weekly = df_week.iloc[idx_start_week:idx_end_week, :]
    df_tracking_data_weekly = df_week.iloc[idx_start_week:idx_end_week, :][utils.momentum.INDICATORS]
    df_tracking_data_weekly['cpct'] = (df_tracking_data_weekly['c'] - df_tracking_data_weekly['c'].iloc[idx_week_offset-1]) / df_tracking_data_weekly['c'].iloc[idx_week_offset-1] * 100
    df_tracking_data_weekly.columns = [f'w_{col}' for col in df_tracking_data_weekly.columns]

    if not any(df_spy_day.index == reaction_date):
      print(
        f'No SPY data for {ticker} at {reaction_date} (idx {idx}). Skipping gap evaluation.'
      )
      # raise("No spy data")
      continue
    idx_spy = df_spy_day.index.get_loc(reaction_date)
    spy_ref_value = df_spy_day['c'].iloc[idx_spy - OFFSET_DAYS]
    df_tracking_data[f'spy'] = (df_spy_day['c'].iloc[idx_spy - OFFSET_DAYS:idx_spy + OFFSET_DAYS] - spy_ref_value) / spy_ref_value * 100

    idx_spy_week = df_spy_week.index.get_loc(df_week.index[idx_week])
    idx_spy_week_start = idx_spy_week - idx_week_offset if idx_spy_week - idx_week_offset >= 0 else 0
    idx_spy_week_end = idx_spy_week + idx_end_week_offset if idx_spy_week + idx_end_week_offset <= len(df_spy_week) - 1 else len(df_spy_week) - 1
    spy_ref_value_week = df_spy_week['c'].iloc[idx_spy_week]
    df_tracking_data_weekly['w_spy'] = (df_spy_week['c'].iloc[idx_spy_week_start:idx_spy_week_end] - spy_ref_value_week) / spy_ref_value_week * 100

    vlines = dict(vlines=reaction_date, colors= ['darkviolet'], linewidths=[0.4], linestyle=['--'])
    fig_day = utils.plots.plot_multi_pane_mpl(df_plotting_data,f'{ticker} {reaction_date}', df_spy_day.iloc[idx_spy - OFFSET_DAYS:idx_spy + OFFSET_DAYS], vlines)
    fig_day.savefig(f'finance/_data/peads/plots/{ticker}/{reaction_date.date()}_D.png')
    vlines_week = dict(vlines=df_week.iloc[idx_week].name, colors= ['darkviolet'], linewidths=[0.4], linestyle=['--'])
    fig_week = utils.plots.plot_multi_pane_mpl(df_plotting_data_weekly, f'{ticker} {reaction_date}', df_spy_week.iloc[idx_spy_week_start:idx_spy_week_end], vlines_week)
    fig_week.savefig(f'finance/_data/peads/plots/{ticker}/{reaction_date.date()}_W.png')
    plt.close('all')
    #%%
    peads_eval = {'gappct': df_peds.gappct}

    if swing_data.market_cap is not None and swing_data.market_cap.empty is False:
      nearest_mc = swing_data.market_cap.iloc[swing_data.market_cap.index.get_indexer([reaction_date], method='nearest')[0]]
      peads_eval['market_cap'] = nearest_mc['market_cap']
      peads_eval['market_cap_date'] = nearest_mc.name

    # Elegant flattening:
    # 1. Set index to the desired suffixes (-1, 0, 1...)
    df_tracking_data.index = np.arange(len(df_tracking_data)) - OFFSET_DAYS
    df_tracking_data_weekly.index = np.arange(len(df_tracking_data_weekly)) - OFFSET_WEEKS

    # 2. Unstack turns it into a Series with a MultiIndex (Column, Suffix)
    # 3. Use a dictionary comprehension to join them into 'c-1', 'rvol0', etc.
    tracking_data_dict = {f"{col}{idx}": val for (col, idx), val in df_tracking_data.unstack().items()}
    tracking_data_dict_weekly = {f"{col}{idx}": val for (col, idx), val in df_tracking_data_weekly.unstack().items()}
    flat_metrics = {**earnings_event.to_dict(), **tracking_data_dict, **tracking_data_dict_weekly}

    peads_eval.update(flat_metrics)
    peads_data.append(peads_eval)
    #%%
  df_peads_stats = pd.DataFrame(peads_data)

  # %%
  df_peads_stats.to_pickle(f'finance/_data/peads/{ticker}.pkl')

# %% Combine all gaps
ticker = liquid_stocks[42]
# start_at = liquid_symbols.index('AGG')
start_at = 0
peads_data = []
for ticker in liquid_stocks[start_at:]:
  # %%
  df_ticker = pd.read_pickle(f'finance/_data/peads/{ticker}.pkl')

  print(f'Processing {ticker}...')
  if df_ticker.empty: continue
  peads_data.append(df_ticker)

## %%
df_peads = pd.concat(peads_data)
df_peads.to_pickle(f'finance/_data/all_peads.pkl')
