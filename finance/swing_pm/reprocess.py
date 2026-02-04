# %%
import os
import pickle
from datetime import timedelta, datetime

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

# spy_data = utils.swing_trading_data.SwingTradingData('SPY', offline=True)
# data = utils.swing_trading_data.SwingTradingData('DCOM', offline=True)
# utils.plots.plot_pyqtgraph(spy_data.df_day, display_range=51)
df_market_cap = pd.read_csv('finance/_data/MarketCapThresholds.csv')

name = 'atr_x'
# name = 'std_x'
# name = 'peads'

##%%
def classify_market_cap(mcap_value, year, df_thresholds):
  """Classifies market cap based on historical thresholds."""
  if mcap_value is None or np.isnan(mcap_value):
    return "Unknown"

  # Get thresholds for the closest available year
  year_idx = df_thresholds['Year'].sub(year).abs().idxmin()
  row = df_thresholds.loc[year_idx]

  if mcap_value >= row['Large-Cap Entry (S&P 500)']:
    return "Large-Cap"
  elif mcap_value >= row['Mid-Cap Entry (S&P 400)']:
    return "Mid-Cap"
  elif mcap_value >= row['Small-Cap Entry (Russell 2000)']:
    return "Small-Cap"
  else:
    return "Micro-Cap"

##%%
def calculate_performance(df_ticker, df_day, length_days):
  df_c = df_day.iloc[df_day.index.get_indexer(df_ticker.date - timedelta(days=length_days), method='ffill')]['c'].copy()
  df_diff = df_ticker.date - df_c.index
  df_c[(df_diff.abs() > timedelta(days=length_days + 5)).values] = np.nan

  return df_c
# df_atr2xs = pd.concat(atr2x_data)
# df_atr2xs.to_pickle(f'finance/_data/all_atr2x.pkl')

# %% Create plots
tickers = {'atr_x': 'LNN', 'std_x': 'BCH', 'peads': 'NA'}
start_at = 0
ticker = tickers[name]
start_at = liquid_symbols.index(ticker)
# atr2x_data = []
for ticker in liquid_symbols[start_at:]:
  # %%
  print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Processing {ticker} - {liquid_symbols.index(ticker)} of {len(liquid_symbols)} ...')
  filename = f'finance/_data/{name}/{ticker}.pkl'
  if not os.path.exists(filename): continue
  df_ticker = pd.read_pickle(filename)
  if df_ticker.empty: continue
  swing_data = utils.swing_trading_data.SwingTradingData(ticker, offline=True, metainfo=False)
  df_day = swing_data.df_day
  df_week = swing_data.df_week
  #%% Calculate performance
  if name in ['peads']:
    for tf_name, tf_days in [('1M', 30), ('3M', 90), ('6M', 180)]:
      df = calculate_performance(df_ticker, df_day, tf_days)
      df_ticker[tf_name] = df.values
      df_ticker[tf_name + '_chg'] = utils.pct.percentage_change_array(df_ticker['c-1'], df.values) # (df_ticker['c-1'] - df_1M.values) / df_1M.values * 100
    df_ticker.to_pickle(filename)

  if 'market_cap' not in df_ticker.columns:
    swing_data = utils.swing_trading_data.SwingTradingData(ticker, offline=True)
    if swing_data.market_cap is not None and not swing_data.market_cap.empty:
      mkp_idx = swing_data.market_cap.index.get_indexer(df_ticker.date, method="nearest")
      df_ticker['market_cap'] = swing_data.market_cap.iloc[mkp_idx]['market_cap'].values

  # --- New Logic: Classify and Save Market Cap ---
  if 'market_cap' in df_ticker.columns:
    df_ticker['mcap_class'] = df_ticker.apply(
      lambda row: classify_market_cap(row['market_cap'], row.date.year, df_market_cap),
      axis=1
    )
    df_ticker.to_pickle(filename)
  #%%
  # i, row = next(df_ticker.iterrows())
  plot_path = f'finance/_data/{name}/plots/{ticker}'
  os.makedirs(plot_path, exist_ok=True)
  num_rows = len(df_ticker)
  for idx_row, (i, row) in enumerate(df_ticker.iterrows(), 1):
    start_time = datetime.now()
    max_date = df_day.index.get_indexer([row.date], method='nearest')[0] + 25
    file_basename = f'{plot_path}/{row.date.date()}'

    # Build dynamic title with classification and performance
    mcap_cat = row.get('mcap_class', 'Unknown')
    perf_str = f"1M: {row['1M_chg']:.1f}% | 3M: {row['3M_chg']:.1f}% | 6M: {row['6M_chg']:.1f}%"
    if 'eps' in row.index and row['eps'] is not None and not pd.isna(row['eps']): perf_str += f" | EPS: {row['eps']:.2f}"
    if 'eps_est' in row.index and row['eps_est'] is not None and not pd.isna(row['eps_est']): perf_str += f" | EPS EST: {row['eps_est']:.2f}"
    full_title = f"{ticker} ({mcap_cat}) - {row.date.date()} | {perf_str}"

    # Daily Plot Timing
    t0_d = datetime.now()
    utils.plots.export_swing_plot(df_day.iloc[max_date-25:max_date], path=f'{file_basename}_D.png', vlines=[row.date], display_range=51, width=1920, height=1080, title=full_title)
    # utils.plots.interactive_swing_plot(df_day.iloc[:max_date], display_range=51)
    dur_d = (datetime.now() - t0_d).total_seconds()

    # Weekly Plot Timing
    max_date_w_idx = df_week.index.get_indexer([row.date], method='ffill')[0]
    max_date_w = df_week.index[max_date_w_idx]
    t0_w = datetime.now()
    utils.plots.export_swing_plot(df_week.iloc[max_date_w_idx-8:max_date_w_idx + 8], path=f'{file_basename}_W.png',
                                  vlines=[max_date_w], display_range=17, width=1920, height=1080, title=full_title)

    dur_w = (datetime.now() - t0_w).total_seconds()

    print(f'  - [{idx_row}/{num_rows}] {row.date.date()}: Daily {dur_d:.2f}s, Weekly {dur_w:.2f}s (Total: {(datetime.now() - start_time).total_seconds():.2f}s)')

  # Heavy cleanup after each ticker to reset memory fragmentation
  import gc
  gc.collect()
  utils.plots._GLOBAL_QT_APP.processEvents()

#%% Create the combined files to analyze
# for name in ['atr_x', 'std_x', 'peads']:
for name in ['std_x']:
  data = []
  for ticker in liquid_symbols:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Processing {ticker} - {liquid_symbols.index(ticker)} of {len(liquid_symbols)} ...')
    filename = f'finance/_data/{name}/{ticker}.pkl'
    if not os.path.exists(filename): continue
    df_ticker = pd.read_pickle(filename)
    data.append(df_ticker)
  df_name = pd.concat(data)
  df_name.to_pickle(f'finance/_data/all_{name}.pkl')

#%% Remove obsolete columns in order to handle sizes
for name in ['atr_x', 'std_x', 'peads']:
  df = pd.read_pickle(f'finance/_data/all_{name}.pkl')



# df_atr2xs = pd.concat(atr2x_data)
# df_atr2xs.to_pickle(f'finance/_data/all_atr2x.pkl')
