import os
import pickle
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import finance.utils as utils

# Constants
INDICATORS = ['c', 'v', 'atrp1', 'atrp9', 'atrp14', 'atrp20', 'atrp50', 'pct', 'rvol50', 'std_mv', 'iv', 'hv9', 'hv14', 'hv20',
              'ema100_dist', 'ema50_dist', 'ema100_dist', 'ema200_dist', 'ema20_dist', 'ema10_dist',
              'ema10_slope', 'ema20_slope', 'ema50_slope', 'ema100_slope', 'ema200_slope']

def classify_market_cap(mcap_value, year, df_thresholds):
    """Classifies market cap based on historical thresholds."""
    if mcap_value is None or np.isnan(mcap_value):
        return "Unknown"

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

def get_earnings_proximity(target_date, df_earnings, days_tolerance=4):
    """Checks if there is an earnings event near the target date."""
    if df_earnings.empty:
        return False
    
    # Check for earnings within tolerance window
    mask = (df_earnings['date'] >= target_date - timedelta(days=days_tolerance)) & \
           (df_earnings['date'] <= target_date + timedelta(days=days_tolerance))
    
    return not df_earnings[mask].empty

#%%
# Setup Paths
output_name = 'ath_run_analysis'
base_path = f'finance/_data/{output_name}'
data_path = f'{base_path}/data'
os.makedirs(data_path, exist_ok=True)

# Load Core Data
print("Loading core data...")
liquid_symbols = pickle.load(open('finance/_data/liquid_symbols.pkl', 'rb'))
df_market_cap_thresholds = pd.read_csv('finance/_data/MarketCapThresholds.csv')

#%%
# Iteration Settings
SKIP = -1
# start_at = 0
start_at = len(liquid_symbols)

symbols_to_process = liquid_symbols[start_at::SKIP]
total_symbols = len(symbols_to_process)
#%%
for i, ticker in enumerate(symbols_to_process):
    ticker_start = time.time()
    
    # Load Earnings Data
    earnings_path = f'finance/_data/earnings_cleaned/{ticker}.csv'
    df_earnings = pd.DataFrame()
    if os.path.exists(earnings_path):
        df_earnings = pd.read_csv(earnings_path)
        df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')

    # Load Price Data
    swing_data = utils.swing_trading_data.SwingTradingData(ticker, offline=True, metainfo=False)
    if swing_data.empty or swing_data.df_week.empty:
        # print(f"  Swing data empty for {ticker}, skipping.")
        continue

    df_week = swing_data.df_week.copy()
    
    # Need at least a few weeks of history
    if len(df_week) < 10:
        continue

    # Load Market Cap History for Lookups
    swing_data_full = utils.swing_trading_data.SwingTradingData(ticker, offline=True)
    ts_market_cap = swing_data_full.market_cap

    events_data = []

    # --- ATH Run Detection Logic ---
    # Variables to track the state
    global_max_h = 0.0
    last_ath_peak_date = pd.NaT
    
    in_run = False
    run_start_idx = -1
    run_peak_high = -1.0
    
    # We iterate starting from index 1 because we need prev_week_low (index 0 is just setup)
    # We initialize global_max_h with the first week's high
    global_max_h = df_week.iloc[0]['h']
    last_ath_peak_date = df_week.index[0]

    for idx in range(1, len(df_week)):
        current_date = df_week.index[idx]
        current_week = df_week.iloc[idx]
        prev_week = df_week.iloc[idx-1]
        
        current_h = current_week['h']
        current_c = current_week['c']
        prev_l = prev_week['l']
        
        # Check for New ATH
        is_new_ath = current_h > global_max_h
        
        if in_run:
            # UPDATE RUN STATE
            if current_h > run_peak_high:
                run_peak_high = current_h
            
            # CHECK EXIT CONDITION
            # "close of the weekly is not below the low of the prior week"
            # Breakdown occurs if close < prev_l
            if current_c < prev_l:
                # RUN ENDS
                in_run = False
                run_end_date = current_date
                
                # Metrics
                run_start_date = df_week.index[run_start_idx]
                
                # Length: number of weeks the run lasted.
                # If start=idx 5, end=idx 10 (breakdown), length is usually considered the duration
                # or the count of qualifying candles. 
                # Let's use inclusive count of weeks in the run before breakdown + breakdown week?
                # Usually simple diff of indices matches "weeks duration".
                run_length_weeks = idx - run_start_idx
                
                # Severity of downmove
                # (Run Peak - Breakdown Close) / Run Peak
                # This measures how much value was lost from top to the close that confirmed the end.
                if run_peak_high > 0:
                    severity = (run_peak_high - current_c) / run_peak_high
                else:
                    severity = 0.0
                
                # Earnings Correlation
                start_with_earnings = get_earnings_proximity(run_start_date, df_earnings)
                end_with_earnings = get_earnings_proximity(run_end_date, df_earnings)
                
                # Indicators at START of run
                start_indicators = df_week.iloc[run_start_idx][INDICATORS].to_dict()
                # Rename keys to w_{key} to indicate weekly source
                start_indicators = {f"w_{k}": v for k, v in start_indicators.items()}
                
                # Market Cap at Start
                mcap_val = np.nan
                if ts_market_cap is not None and not ts_market_cap.empty:
                    mkp_idx = ts_market_cap.index.get_indexer([run_start_date], method="nearest")[0]
                    mcap_val = ts_market_cap.iloc[mkp_idx]['market_cap']

                # Build Event Row
                row = {
                    'symbol': ticker,
                    'run_start_date': run_start_date,
                    'run_end_date': run_end_date,
                    'run_length_weeks': run_length_weeks,
                    'severity': severity,
                    'run_peak_high': run_peak_high,
                    'weeks_since_last_ath': current_run_dist_weeks,
                    'start_with_earnings': start_with_earnings,
                    'end_with_earnings': end_with_earnings,
                    'market_cap': mcap_val,
                    'mcap_class': classify_market_cap(mcap_val, run_start_date.year, df_market_cap_thresholds)
                }
                # Add indicators
                row.update(start_indicators)
                
                events_data.append(row)
        else:
            # NOT IN RUN
            if is_new_ath:
                # Potential Start of Run
                # Check if it immediately fails the condition?
                # Condition: "close of the weekly is not below the low of the prior week"
                # If current_c < prev_l, the breakout fails immediately.
                if current_c >= prev_l:
                    in_run = True
                    run_start_idx = idx
                    run_peak_high = current_h
                    
                    # Calculate distance to LAST ATH (before this run started)
                    if pd.notna(last_ath_peak_date):
                        # Calculate weeks diff
                        diff = (current_date - last_ath_peak_date).days / 7
                        current_run_dist_weeks = diff
                    else:
                        current_run_dist_weeks = np.nan
        
        # Update Global History Tracker
        # We update this regardless of run state to track "All Time High"
        if current_h > global_max_h:
            global_max_h = current_h
            last_ath_peak_date = current_date # The date the new ATH was set

    if not events_data:
        # print(f"  No completed runs for {ticker}")
        continue
    
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {ticker}: Found {len(events_data)} runs.')

    df_ticker_events = pd.DataFrame(events_data)
    
    # Save Pickled Data
    ticker_file = f'{data_path}/{ticker}.pkl'
    df_ticker_events.to_pickle(ticker_file)

#%%
# --- Final Aggregation ---
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
