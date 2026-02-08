import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

import finance.utils as utils

def create_plots():
    # Setup Paths
    output_name = 'momentum_earnings'
    base_path = f'finance/_data/{output_name}'
    plot_path = f'{base_path}/plots'
    data_path = f'{base_path}/data'
    os.makedirs(plot_path, exist_ok=True)
    
    # Check for data directory
    if not os.path.exists(data_path):
        print(f"Data directory {data_path} not found. Please run the data creation script first.")
        return

    # Process all pickle files in the data directory
    files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    total_files = len(files)
    
    print(f"Found {total_files} data files to process.")

    # Plotting offsets
    PLOT_DAYS = 100
    PLOT_WEEKS = 50

    for i, filename in enumerate(files):
        ticker = filename.replace('.pkl', '')
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Plotting {i+1}/{total_files}: {ticker}...')
        
        t0 = time.time()

        # Load Event Data
        try:
            df_ticker_events = pd.read_pickle(os.path.join(data_path, filename))
        except Exception as e:
            print(f"  Error loading data for {ticker}: {e}")
            continue

        if df_ticker_events.empty:
            continue

        # Load Swing Data (OHLCV needed for context)
        # using offline=True as in the original script to avoid DB calls if possible
        try:
            swing_data = utils.swing_trading_data.SwingTradingData(ticker, offline=True, metainfo=False)
            if swing_data.empty:
                print(f"  Swing data empty for {ticker}, skipping.")
                continue
            df_day = swing_data.df_day
            df_week = swing_data.df_week
        except Exception as e:
            print(f"  Error loading swing data for {ticker}: {e}")
            continue

        # Prepare Plot Directory
        ticker_plot_path = f'{plot_path}/{ticker}'
        os.makedirs(ticker_plot_path, exist_ok=True)

        # Plotting Loop
        for idx_row, (i_evt, row) in enumerate(df_ticker_events.iterrows()):
            try:
                # Event index in daily data
                idx_day_arr = df_day.index.get_indexer([row.date], method='nearest')
                if len(idx_day_arr) == 0:
                    continue
                idx_day = idx_day_arr[0]

                file_basename = f'{ticker_plot_path}/{row.date.date()}'

                # Dynamic Title
                mcap_cat = row.get('mcap_class', 'Unknown')
                perf_str = f"1M: {row['1M_chg']:.1f}% | 3M: {row['3M_chg']:.1f}% | 6M: {row['6M_chg']:.1f}%"
                if not pd.isna(row['eps']): perf_str += f" | EPS: {row['eps']:.2f}"
                if not pd.isna(row['eps_est']): perf_str += f" | Est: {row['eps_est']:.2f}"

                full_title = f"{ticker} ({mcap_cat}) - {row.date.date()} | {perf_str}"

                # Daily Plot (Last 100 days, Next 100 days)
                d_start = max(0, idx_day - PLOT_DAYS)
                d_end = min(len(df_day), idx_day + PLOT_DAYS + 1)
                slice_day = df_day.iloc[d_start:d_end]

                utils.plots.export_swing_plot(
                    slice_day,
                    path=f'{file_basename}_D.png',
                    vlines=[row.date],
                    display_range=len(slice_day),
                    width=1920, height=1080,
                    title=full_title
                )

                # Weekly Plot (Last 50 weeks, Next 50 weeks)
                idx_week_arr = df_week.index.get_indexer([row.date], method='ffill')
                if len(idx_week_arr) == 0:
                    continue
                idx_week = idx_week_arr[0]
                
                w_start = max(0, idx_week - PLOT_WEEKS)
                w_end = min(len(df_week), idx_week + PLOT_WEEKS + 1)
                slice_week = df_week.iloc[w_start:w_end]

                # The vline needs to be on the weekly index
                week_vline_date = df_week.index[idx_week]

                utils.plots.export_swing_plot(
                    slice_week,
                    path=f'{file_basename}_W.png',
                    vlines=[week_vline_date],
                    display_range=len(slice_week),
                    width=1920, height=1080,
                    title=full_title
                )
            except Exception as e:
                print(f"  Error generating plot for event {row.date}: {e}")
                continue

        duration = time.time() - t0
        print(f"  Finished {ticker} in {duration:.2f}s")

if __name__ == '__main__':
    create_plots()
