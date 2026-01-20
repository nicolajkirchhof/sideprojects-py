# %%
import os
import time
from datetime import datetime

import numpy as np

import pandas as pd

import finance.utils as utils

INDICATORS = ['c', 'v', 'atrp9', 'atrp14', 'atrp20', 'ac100_lag_1', 'ac100_lag_5', 'ac100_lag_20', 'ac_comp', 'ac_mom', 'ac_mr',
              'ac_inst', 'pct', 'rvol20', 'rvol50', 'std_mv', 'iv', 'hv9', 'hv14', 'hv20', 'hv50', 'ema20_dist', 'ema10_dist', 'ema10_slope', 'ema20_slope']

# def statid_plot_swing_data():
#   plot_start = time.time()
#   vlines = dict(vlines=df_momentum.name, colors= ['darkviolet'], linewidths=[0.4], linestyle=['--'])
#   fig_day = utils.plots.plot_multi_pane_mpl(df_plotting_data,f'{ticker} {df_momentum.name.date()}', df_spy_day.iloc[idx_spy - offset_days:idx_spy + offset_days], vlines, fig=fig_day)
#   fig_day.savefig(f'finance/_data/{name}/plots/{ticker}/{df_momentum.name.date()}_D.png')
#   vlines_week = dict(vlines=df_week.iloc[idx_week].name, colors= ['darkviolet'], linewidths=[0.4], linestyle=['--'])
#   fig_week = utils.plots.plot_multi_pane_mpl(df_plotting_data_weekly, f'{ticker} {df_momentum.name.date()}', df_spy_week.iloc[idx_spy_week_start:idx_spy_week_end], vlines_week, fig_week)
#   fig_week.savefig(f'finance/_data/{name}/plots/{ticker}/{df_momentum.name.date()}_W.png')
#   plot_duration = time.time() - plot_start

def evaluate_liquid_symbols(name, fn_flt, spy_data, liquid_symbols, offline=True, offset_days=25, offset_weeks=8, indicators=None, start_at_name=None, stop_at_name=None):
  if fn_flt is None:
    raise ValueError('fn_flt must be provided.')

  if spy_data is None or spy_data.empty:
    raise ValueError('spy_data must be provided.')

  if indicators is None:
    indicators = INDICATORS

  df_spy_day = spy_data.df_day
  df_spy_week = spy_data.df_week
  fig_day = None
  fig_week = None

  # %%
  # ticker = df_liquid_symbols[0]
  start_at = 0
  if start_at_name is not None:
    start_at = liquid_symbols.index(start_at_name)
  # TO DO AGAIN RGEN
  # start_at = df_liquid_symbols[df_liquid_symbols.Symbol == 'OUT'].index[0] + 1
  end_at = len(liquid_symbols)
  if stop_at_name is not None:
    end_at = liquid_symbols.index(stop_at_name) + 1
  ##%%
  for ticker in liquid_symbols[start_at:end_at]:
    ticker_start_time = time.time()
    if os.path.exists(f'finance/_data/ibkr/stk/{ticker}.pkl'): continue
    print(f'{datetime.now()} Processing {ticker}...')
    # %%

    os.makedirs(f'finance/_data/{name}/plots/{ticker}', exist_ok=True)
    df_earnings = pd.read_csv(f'finance/_data/earnings_cleaned/{ticker}.csv')
    df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')

    load_start = time.time()
    swing_data = utils.swing_trading_data.SwingTradingData(ticker, offline=offline)
    if swing_data.empty or swing_data.info is None: continue
    df_day = swing_data.df_day
    df_week = swing_data.df_week
    # print(f"  [DEBUG] Data loading for {ticker}: {time.time() - load_start:.2f}s")
    # %% Dataset of gaps: gap%, after 3 days, after 6 days, after 9 days, after 14 days
    momentums = df_day[fn_flt(df_day)]

    momentum_data = []
    for idx, df_momentum in momentums.iterrows():
      event_start = time.time()
      idx = df_day.index.get_loc(df_momentum.name)
      if idx < offset_days: continue
      df_plotting_data = df_day.iloc[idx - offset_days:idx + offset_days, :]
      df_tracking_data = df_day.iloc[idx - offset_days:idx + offset_days, :][INDICATORS]
      if len(df_tracking_data) < 2 * offset_days:
        print(f'Tracking data too short for {ticker} at {df_momentum.name}. Skipping earnings evaluation.')
        # raise('tracking data too short')
        continue
      df_tracking_data['cpct'] = (df_tracking_data['c'] - df_tracking_data['c'].iloc[offset_days-1]) / df_tracking_data['c'].iloc[offset_days-1] * 100

      idx_week = df_week.index.get_indexer([df_momentum.name], method='ffill')[0]
      idx_week_offset = offset_weeks if idx_week >= offset_weeks else idx_week
      idx_start_week = idx_week - offset_weeks if idx_week - offset_weeks >= 0 else 0
      idx_end_week = idx_week + offset_weeks if idx_week + offset_weeks <= len(df_week) - 1 else len(df_week) - 1
      idx_end_week_offset = idx_end_week-idx_start_week-idx_week_offset
      df_plotting_data_weekly = df_week.iloc[idx_start_week:idx_end_week, :]
      df_tracking_data_weekly = df_week.iloc[idx_start_week:idx_end_week, :][INDICATORS]
      df_tracking_data_weekly['cpct'] = (df_tracking_data_weekly['c'] - df_tracking_data_weekly['c'].iloc[idx_week_offset-1]) / df_tracking_data_weekly['c'].iloc[idx_week_offset-1] * 100
      df_tracking_data_weekly.columns = [f'w_{col}' for col in df_tracking_data_weekly.columns]

      if not any(df_spy_day.index == df_momentum.name):
        print(
          f'No SPY data for {ticker} at {df_momentum.name} (idx {idx}). Skipping gap evaluation.'
        )
        # raise("No spy data")
        continue
      idx_spy = df_spy_day.index.get_loc(df_momentum.name)

      spy_ref_value = df_spy_day['c'].iloc[idx_spy - offset_days]
      df_tracking_data[f'spy'] = (df_spy_day['c'].iloc[idx_spy - offset_days:idx_spy + offset_days] - spy_ref_value) / spy_ref_value * 100

      idx_spy_week = df_spy_week.index.get_loc(df_week.index[idx_week])
      idx_spy_week_start = idx_spy_week - idx_week_offset if idx_spy_week - idx_week_offset >= 0 else 0
      idx_spy_week_end = idx_spy_week + idx_end_week_offset if idx_spy_week + idx_end_week_offset <= len(df_spy_week) - 1 else len(df_spy_week) - 1
      spy_ref_value_week = df_spy_week['c'].iloc[idx_spy_week]
      df_tracking_data_weekly['w_spy'] = (df_spy_week['c'].iloc[idx_spy_week_start:idx_spy_week_end] - spy_ref_value_week) / spy_ref_value_week * 100
      #%%
      momentum_eval = {'gappct': df_momentum.gappct}

      if swing_data.market_cap is not None and swing_data.market_cap.empty is False:
        nearest_mc = swing_data.market_cap.iloc[swing_data.market_cap.index.get_indexer([df_momentum.name], method='nearest')[0]]
        momentum_eval['market_cap'] = nearest_mc['market_cap']
        momentum_eval['market_cap_date'] = nearest_mc.name

      postEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx - 1]), 'when']
      preEarnings = df_earnings.loc[df_earnings.date.eq(df_day.index[idx]), 'when']
      hasEarnings = postEarnings.iat[0] == 'post' if not postEarnings.empty else preEarnings.iat[
                                                                                 0] == 'pre' if not preEarnings.empty else False
      momentum_eval = {'symbol': ticker, 'earnings': hasEarnings, 'date': df_momentum.name,
                    'gappct': df_momentum.gappct, 'c': df_momentum.c,
                    'is_etf': swing_data.info.is_etf, 'atrp20': df_momentum.atrp20}

      # Elegant flattening:
      # 1. Set index to the desired suffixes (-1, 0, 1...)
      df_tracking_data.index = np.arange(len(df_tracking_data)) - offset_days
      df_tracking_data_weekly.index = np.arange(len(df_tracking_data_weekly)) - offset_weeks

      # 2. Unstack turns it into a Series with a MultiIndex (Column, Suffix)
      # 3. Use a dictionary comprehension to join them into 'c-1', 'rvol0', etc.
      tracking_data_dict = {f"{col}{idx}": val for (col, idx), val in df_tracking_data.unstack().items()}
      tracking_data_dict_weekly = {f"{col}{idx}": val for (col, idx), val in df_tracking_data_weekly.unstack().items()}
      flat_metrics = {**momentum_eval, **tracking_data_dict, **tracking_data_dict_weekly}

      momentum_eval.update(flat_metrics)
      momentum_data.append(momentum_eval)
      # print(f"    [DEBUG] Event {df_momentum.name.date()} processed in {time.time() - event_start:.2f}s)")
    #%%
    df_momentum_data = pd.DataFrame(momentum_data)

    # %%
    df_momentum_data.to_pickle(f'finance/_data/{name}/{ticker}.pkl')
    # print(f"  [DEBUG] Total ticker {ticker} time: {time.time() - ticker_start_time:.2f}s")

