import os
import pickle
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import finance.utils as utils

# Constants from momentum.py
INDICATORS = ['c', 'v', 'atrp1', 'atrp9', 'atrp14', 'atrp20', 'atrp50', 'pct', 'rvol50', 'std_mv', 'iv', 'hv9', 'hv14', 'hv20',
              'ema200_dist', 'ema100_dist', 'ema50_dist', 'ema20_dist', 'ema10_dist', 'ema5_dist',
              'ema5_slope', 'ema10_slope', 'ema20_slope', 'ema50_slope', 'ema100_slope', 'ema200_slope']
SPY_INDICATORS = ['hv9', 'hv14', 'hv20', 
                  'ema10_dist', 'ema20_dist', 'ema50_dist', 'ema100_dist', 'ema200_dist',
                  'ema10_slope', 'ema20_slope', 'ema50_slope', 'ema100_slope', 'ema200_slope']
OFFSET_DAYS = 25
OFFSET_WEEKS = 8

#%%
# The dataset has the following columns 'symbol', 'date', 'is_earnings', 'event_type', 'eps', 'eps_est',
#   'earnings_when', 'gappct', 'market_cap', 'market_cap_date', 'market_cap_class'
#   '1M', '1M_chg', '3M', '3M_chg', '6M', '6M_chg', '12M', '12M_chg'
# In addition, the following columns track changes before and after the event they are tracked
# as {name}XX and w_{name}XX for daily and weekly values before and after the event.
#   Daily columns are from -25 to 24 whereas 0 is the event day
#   Weekly columns are from -8 to 8 whereas 0 is the event week
# Tracked names
#   'c' => close, 'spy' => spy changes (relative to window start), 'v' => volume,
#   'atrp1/9/14/20/50' => ATR percentage, 'pct' => Percent Change, 'rvol50' => Relative Volatility 50-day,
#   'std_mv' => 20 day standard deviation, 'iv' => implied volatility,
#   'hv9/14/20' => Historical Volatility 9/14/20-day,
#   'ema10/20/50/100/200_dist' => Distance to EMA 10/20/50/100/200,
#   'ema10/20/50/100/200_slope' => Slope of EMA 10/20/50/100/200,
#   'cpct' => Percentage change in reference to the day before the event (c-1)
#   'spy_{indicator}' => SPY equivalent for specific indicators (hv, ema_dist, ema_slope)

def classify_market_cap(mcap_value, year, df_thresholds):
    """Classifies market cap based on historical thresholds (from reprocess.py)."""
    if mcap_value is None or np.isnan(mcap_value):
        return "Unknown"

    # Get thresholds for the closest available year
    year_idx = df_thresholds['Year'].sub(year).abs().idxmin()
    row = df_thresholds.loc[year_idx]

    if mcap_value >= row['Large-Cap Entry (S&P 500)']:
        return "Large"
    elif mcap_value >= row['Mid-Cap Entry (S&P 400)']:
        return "Mid"
    elif mcap_value >= row['Small-Cap Entry (Russell 2000)']:
        return "Small"
    else:
        return "Micro"

def calculate_performance(df_ticker, df_day, length_days):
  df_c = df_day.iloc[df_day.index.get_indexer(df_ticker.date - timedelta(days=length_days), method='ffill')]['c'].copy()
  df_diff = df_ticker.date - df_c.index
  df_c[(df_diff.abs() > timedelta(days=length_days + 5)).values] = np.nan

  return df_c

#%%
# Setup Paths
output_name = 'momentum_earnings'
base_path = f'finance/_data/{output_name}'
plot_path = f'{base_path}/plots'
data_path = f'{base_path}/data'
os.makedirs(data_path, exist_ok=True)

# Load Core Data
print("Loading core data...")
liquid_symbols = pickle.load(open('finance/_data/liquid_symbols.pkl', 'rb'))
df_market_cap_thresholds = pd.read_csv('finance/_data/MarketCapThresholds.csv')

spy_data = utils.swing_trading_data.SwingTradingData('SPY', offline=True)
df_spy_day = spy_data.df_day
df_spy_week = spy_data.df_week

#%%
# Iteration Settings
# SKIP = 100
SKIP = 1
start_at = 0
# start_at = len(liquid_symbols)
# start_at = liquid_symbols.index('UHA.B') # Debugging start point
# ticker = liquid_symbols[start_at]
# symbols_to_process = ['MSFT']
# symbols_to_process = ['IWM']
symbols_to_process = liquid_symbols[start_at::SKIP]
total_symbols = len(symbols_to_process)
#%%
for i, ticker in enumerate(symbols_to_process):
    ticker_start = time.time()
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Processing {i+1}/{total_symbols}: {ticker}...')

    # Time: Earnings Load
    t0 = time.time()
    earnings_path = f'finance/_data/earnings_cleaned/{ticker}.csv'
    if not os.path.exists(earnings_path):
        print(f"  No earnings data for {ticker}, skipping.")
        continue

    df_earnings = pd.read_csv(earnings_path)
    df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')
    t_earnings = time.time() - t0

    # Time: Market Cap Load
    t0 = time.time()
    # Load Market Cap History for Lookups
    # Note: Creating SwingTradingData again without metainfo=False triggers full DB load if not cached/offline
    swing_data_full = utils.swing_trading_data.SwingTradingData(ticker, offline=True)
    if swing_data_full.empty:
        print(f"  No data for {ticker}, skipping.")
        continue

    is_etf = bool(swing_data_full.info.is_etf)
    ts_market_cap = swing_data_full.market_cap
    df_day = swing_data_full.df_day
    df_week = swing_data_full.df_week
    t_mcap = time.time() - t0

    events_map = {}

    # Time: Event Detection
    t0 = time.time()
    # --- 1. Identify Earnings Events ---
    for i_earn, earnings_event in df_earnings.iterrows():
        # Range Check against announcement date
        if df_day.index.min() > earnings_event.date - timedelta(days=OFFSET_DAYS) or \
           df_day.index.max() < earnings_event.date + timedelta(days=OFFSET_DAYS):
            # print(f"  Date {earnings_event.date.date()} out of range for {ticker}, skipping.")
            continue

        # Determine Reaction Date
        # Use get_indexer for safe lookup
        idx_arr = df_day.index.get_indexer([earnings_event.date], method='nearest')
        if len(idx_arr) == 0:
            print(f"  Could not find nearest date for earnings on {earnings_event.date.date()}, skipping.")
            continue
        idx = idx_arr[0]
    
        reaction_idx = idx + 1 if earnings_event.when == 'post' else idx

        # Boundary Check
        if reaction_idx < OFFSET_DAYS or reaction_idx >= len(df_day) - OFFSET_DAYS:
            # print(f"  Earnings reaction index {reaction_idx} out of bounds for {ticker}, skipping.")
            continue

        reaction_date = df_day.iloc[reaction_idx].name
    
        # Store in map (Earnings take priority)
        events_map[reaction_date] = {
            'reaction_idx': reaction_idx,
            'eps': earnings_event.eps,
            'eps_est': earnings_event.eps_est,
            'earnings_when': earnings_event.when,
            'is_earnings': True,
            'event_type': 'earnings'
        }

    # --- 2. Identify ATRP Breakout Events ---
    # Condition: 1.5 * ATRP20 < |PCT|
    if 'atrp20' in df_day.columns and 'pct' in df_day.columns:
        # Create a mask for the condition
        atrp_condition = (1.5 * df_day['atrp20'] < df_day['pct'].abs())
    
        # Iterate through breakout dates
        for reaction_date in df_day.index[atrp_condition]:
            # If date already exists as an earnings event, skip (Earnings priority)
            if reaction_date in events_map:
                continue

            reaction_idx = df_day.index.get_loc(reaction_date)

            # Boundary Check
            if reaction_idx < OFFSET_DAYS or reaction_idx >= len(df_day) - OFFSET_DAYS:
                continue

            events_map[reaction_date] = {
                'reaction_idx': reaction_idx,
                'eps': np.nan,
                'eps_est': np.nan,
                'earnings_when': np.nan,
                'is_earnings': False,
                'event_type': 'atrp_breakout'
            }

    if not events_map:
        # print(f"  No valid events found for {ticker}, skipping.")
        continue

    events_data = []
    t_detection = time.time() - t0

    # Time: Processing Loop
    t0 = time.time()
    # --- 3. Unified Processing Loop ---
    # Process events in chronological order
    for reaction_date in sorted(events_map.keys()):
        meta = events_map[reaction_date]
        reaction_idx = meta['reaction_idx']

        # Slice DataFrames
        df_slice_day = df_day.iloc[reaction_idx - OFFSET_DAYS : reaction_idx + OFFSET_DAYS]
        df_tracking_day = df_slice_day[INDICATORS].copy()
    
        if len(df_tracking_day) < 2 * OFFSET_DAYS or df_slice_day.c.isna().all():
            # print(f"  Insufficient tracking data ({len(df_tracking_day)} rows) for {ticker} at {reaction_date.date()}, skipping.")
            continue

        # Normalized close percentage
        ref_c_day = df_tracking_day['c'].iloc[OFFSET_DAYS-1]
        df_tracking_day['cpct'] = (df_tracking_day['c'] - ref_c_day) / ref_c_day * 100

        # Weekly Logic
        idx_week = df_week.index.get_indexer([reaction_date], method='nearest')[0]
        idx_week_offset = OFFSET_WEEKS if idx_week >= OFFSET_WEEKS else idx_week
        idx_start_week = max(0, idx_week - OFFSET_WEEKS)
        idx_end_week = min(len(df_week) - 1, idx_week + OFFSET_WEEKS)
    
        # Adjust offset if near start/end of data
        effective_week_offset = idx_week - idx_start_week

        df_slice_week = df_week.iloc[idx_start_week : idx_end_week]
        df_tracking_week = df_slice_week[INDICATORS].copy()
    
        if not df_tracking_week.empty:
            ref_c_week = df_tracking_week['c'].iloc[effective_week_offset-1] if effective_week_offset > 0 else df_tracking_week['c'].iloc[0]
            df_tracking_week['cpct'] = (df_tracking_week['c'] - ref_c_week) / ref_c_week * 100
            df_tracking_week.columns = [f'w_{col}' for col in df_tracking_week.columns]

        # SPY Comparison Logic
        if not any(df_spy_day.index == reaction_date):
            # print(f"  No SPY data for {ticker} at {reaction_date.date()}, skipping.")
            continue
    
        idx_spy = df_spy_day.index.get_loc(reaction_date)
        spy_ref_val = df_spy_day['c'].iloc[idx_spy - OFFSET_DAYS]
        spy_slice = (df_spy_day['c'].iloc[idx_spy - OFFSET_DAYS : idx_spy + OFFSET_DAYS] - spy_ref_val) / spy_ref_val * 100
        df_tracking_day['spy'] = spy_slice.values if len(spy_slice) == len(df_tracking_day) else np.nan

        # SPY Indicators
        spy_slice_ind = df_spy_day.iloc[idx_spy - OFFSET_DAYS : idx_spy + OFFSET_DAYS]
        if len(spy_slice_ind) == len(df_tracking_day):
            for col in SPY_INDICATORS:
                if col in spy_slice_ind.columns:
                    df_tracking_day[f'spy_{col}'] = spy_slice_ind[col].values
                else:
                    df_tracking_day[f'spy_{col}'] = np.nan

        idx_spy_week = df_spy_week.index.get_indexer([df_week.index[idx_week]], method='nearest')[0]
        spy_week_start = max(0, idx_spy_week - effective_week_offset)
        spy_week_end = min(len(df_spy_week), spy_week_start + len(df_tracking_week))
    
        spy_ref_week = df_spy_week['c'].iloc[idx_spy_week]
        spy_slice_week = (df_spy_week['c'].iloc[spy_week_start:spy_week_end] - spy_ref_week) / spy_ref_week * 100
    
        # Handle slight length mismatches in weekly data
        if len(spy_slice_week) == len(df_tracking_week):
             df_tracking_week['w_spy'] = spy_slice_week.values
        else:
             df_tracking_week['w_spy'] = np.nan

        # --- Flattening (Elegant approach from momentum.py) ---
        # 1. Reindex to relative integers (-25 to +25)
        df_tracking_day.index = np.arange(len(df_tracking_day)) - OFFSET_DAYS
        df_tracking_week.index = np.arange(len(df_tracking_week)) - effective_week_offset

        # 2. Unstack to dictionary
        flat_day = {f"{col}{idx}": val for (col, idx), val in df_tracking_day.unstack().items()}
        flat_week = {f"{col}{idx}": val for (col, idx), val in df_tracking_week.unstack().items()}
    
        # Base Event Data
        event_row = {
            'symbol': ticker,
            'date': reaction_date,
            'is_earnings': meta['is_earnings'],
            'is_etf': is_etf,
            'event_type': meta['event_type'],
            'eps': meta['eps'],
            'eps_est': meta['eps_est'],
            'earnings_when': meta['earnings_when'],
            'gappct': df_day.iloc[reaction_idx].gappct,
            'market_cap': np.nan, # To be filled
            'market_cap_date': pd.NaT,
            'market_cap_class': 'Unknown'
        }

        # Market Cap Lookup
        if ts_market_cap is not None and not ts_market_cap.empty:
            mkp_idx = ts_market_cap.index.get_indexer([reaction_date], method="nearest")[0]
            mc_row = ts_market_cap.iloc[mkp_idx]
            event_row['market_cap'] = mc_row['market_cap']
            event_row['market_cap_date'] = mc_row.name
            event_row['market_cap_class'] = classify_market_cap(mc_row['market_cap'], mc_row.name.date().year, df_market_cap_thresholds)

        # Merge all data
        full_row = {**event_row, **flat_day, **flat_week}
        events_data.append(full_row)

    if not events_data:
        # print(f"  No valid events processed for {ticker}, skipping.")
        continue

    df_ticker_events = pd.DataFrame(events_data)

    # --- 2. Reprocess Logic: Performance & Classification ---

    # Calculate Past Performance (1M, 3M, 6M, 12M)
    for tf_name, tf_days in [('1M', 30), ('3M', 90), ('6M', 180), ('12M', 360)]:
        # Note: calculate_performance returns a Series aligned to df_ticker_events
        future_c = calculate_performance(df_ticker_events, df_day, tf_days)
        df_ticker_events[tf_name] = future_c.values
        # Calculate % Change: (Future - Current) / Current
        # using 'c-1' (day before reaction) or 'c0' (reaction day) as base?
        # reprocess.py uses 'c-1' as base.
        df_ticker_events[tf_name + '_chg'] = utils.pct.percentage_change_array(
            df_ticker_events['c-1'], future_c.values
        )

    # Save Pickled Data
    ticker_file = f'{data_path}/{ticker}.pkl'
    df_ticker_events.to_pickle(ticker_file)
    t_process = time.time() - t0

    total_time = time.time() - ticker_start
    print(f"  [Time] Earnings: {t_earnings:.3f}s, MCap: {t_mcap:.3f}s, Detect: {t_detection:.3f}s, Process: {t_process:.3f}s | Total: {total_time:.3f}s")
#%%
# --- 4. Final Aggregation ---
print("Aggregating all files...")
all_data = []
for ticker in liquid_symbols:
    filename = f'{data_path}/{ticker}.pkl'
    if os.path.exists(filename):
        all_data.append(pd.read_pickle(filename))

if all_data:
    df_all = pd.concat(all_data)
    df_all.to_pickle(f'finance/_data/all_{output_name}.pkl')
    print(f"Complete. Saved to finance/_data/all_{output_name}.pkl")
