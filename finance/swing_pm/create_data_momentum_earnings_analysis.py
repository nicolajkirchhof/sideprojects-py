import os
import pickle
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import finance.utils as utils
from finance.utils import underlyings

# %load_ext autoreload
# %autoreload 2

# Constants from momentum.py
INDICATORS = ['c', 'v', 'atrp1', 'atrp9', 'atrp14', 'atrp20', 'atrp50', 'pct', 'rvol20', 'rvol50', 'std_mv', 'iv', 'hv9', 'hv14', 'hv20', 'hv50',
              'ma200_dist', 'ma100_dist', 'ma50_dist', 'ma20_dist', 'ma10_dist', 'ma5_dist',
              'ma5_slope', 'ma10_slope', 'ma20_slope', 'ma50_slope', 'ma100_slope', 'ma200_slope']
SPY_INDICATORS = ['hv9', 'hv14', 'hv20', 'hv50', 'ma10_dist', 'ma20_dist', 'ma50_dist', 'ma100_dist', 'ma200_dist',
                  'ma10_slope', 'ma20_slope', 'ma50_slope', 'ma100_slope', 'ma200_slope']
OFFSET_DAYS = 25
OFFSET_WEEKS = 8
MIN_VOLUME = 750000

# Setup Paths
output_name = 'momentum_earnings'
base_path = f'finance/_data/{output_name}'
data_path = f'{base_path}/ticker'
os.makedirs(data_path, exist_ok=True)

# Load Core Data
liquid_stocks = underlyings.get_liquid_stocks()
liquid_etfs = underlyings.get_liquid_etfs()
liquid_symbols = liquid_stocks + liquid_etfs

#%%
# The dataset has the following columns 'symbol', 'date', 'is_earnings', 'event_types', 'eps', 'eps_est',
#   'earnings_when', 'gappct', 'market_cap', 'market_cap_date', 'market_cap_class', 'original_price',
#   '1M_chg', '3M_chg', '6M_chg', '12M_chg'
#   'evt_atrp_breakout', 'evt_green_line_breakout', 'evt_bb_lower_touch'
# In addition, the following columns track changes before and after the event they are tracked
# as {name}XX and w_{name}XX for daily and weekly values before and after the event.
#   Daily columns are from -25 to 25 whereas 0 is the event day (Total: 51 days)
#   Weekly columns are from -8 to 8 whereas 0 is the event week (Total: 17 weeks)
# Tracked names
#   'c' => close, 'spy' => spy changes (relative to window start), 'v' => volume,
#   'atrp1/9/14/20/50' => ATR percentage, 'pct' => Percent Change, 'rvol20/50' => Relative Volatility 20/50-day,
#   'std_mv' => 20 day standard deviation, 'iv' => implied volatility,
#   'hv9/14/20/50' => Historical Volatility 9/14/20/50-day,
#   'ma5/10/20/50/100/200_dist' => Distance to MA 5/10/20/50/100/200,
#   'ma5/10/20/50/100/200_slope' => Slope of MA 5/10/20/50/100/200,
#   'cpct' => Percentage change in reference to the day before the event (c-1)
#   'spy_{indicator}' => SPY equivalent for specific indicators (hv, ma_dist, ma_slope)

#%%

print("Loading core data...")
spy_data = utils.SwingTradingData('SPY', datasource='offline')
df_spy_day = spy_data.df_day
df_spy_week = spy_data.df_week

#%%
# Iteration Settings
# SKIP = 100
SKIP = 1
start_at = 0
# start_at = len(liquid_symbols)
# start_at = liquid_symbols.index('RUBI') # Debugging start point
# ticker = liquid_symbols[start_at]
# symbols_to_process = ['MSFT']
# symbols_to_process = ['IWM']
symbols_to_process = liquid_symbols[start_at::SKIP]
total_symbols = len(symbols_to_process)
#%%
for i, ticker in enumerate(symbols_to_process):
    ticker_start = time.time()

    # Time: Earnings Load
    t0 = time.time()
    earnings_path = f'finance/_data/earnings_cleaned/{ticker}.csv'
    if not os.path.exists(earnings_path):
        df_earnings = pd.DataFrame(columns=['date', 'eps', 'eps_est', 'when'])
    else:
        df_earnings = pd.read_csv(earnings_path)
        df_earnings['date'] = pd.to_datetime(df_earnings['date'], format='%Y-%m-%d')
    t_earnings = time.time() - t0

    # Time: Market Cap Load
    t0 = time.time()
    # Load Market Cap History for Lookups
    # Note: Creating SwingTradingData again without metainfo=False triggers full DB load if not cached/offline
    is_etf = ticker in liquid_etfs
    swing_data_full = utils.SwingTradingData(ticker, datasource='offline')
    if swing_data_full.empty:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {i+1:4}/{total_symbols}: {ticker:5} | SKIP: No data")
        continue

    ts_market_cap = swing_data_full.market_cap
    df_day = swing_data_full.df_day
    df_week = swing_data_full.df_week
    t_mcap = time.time() - t0

    events_map = {}
    disregarded_count = 0

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
        idx_arr = df_day.index.get_indexer([earnings_event.date], method='nearest')
        if len(idx_arr) == 0:
            print(f"  Could not find nearest date for earnings on {earnings_event.date.date()}, skipping.")
            continue
        idx = idx_arr[0]

        reaction_idx = idx + 1 if earnings_event.when == 'post' else idx

        # Boundary Check
        if reaction_idx < OFFSET_DAYS or reaction_idx >= len(df_day) - OFFSET_DAYS:
            continue

        reaction_date = df_day.iloc[reaction_idx].name
        reaction_volume = df_day.iloc[reaction_idx].v

        if reaction_volume < MIN_VOLUME:
            disregarded_count += 1
            continue

        meta = events_map.get(reaction_date, {
            'reaction_idx': reaction_idx,
            'eps': np.nan,
            'eps_est': np.nan,
            'earnings_when': np.nan,
            'event_types': set(),
        })

        # If we already had the date (e.g. from ATRP), keep reaction_idx consistent with the date.
        meta['reaction_idx'] = reaction_idx
        meta['eps'] = earnings_event.eps
        meta['eps_est'] = earnings_event.eps_est
        meta['earnings_when'] = earnings_event.when
        meta['event_types'].add('earnings')

        events_map[reaction_date] = meta

    # --- 2. Identify ATRP Breakout Events ---
    # Condition: 1.5 * ATRP20 < |PCT|
    if 'atrp20' in df_day.columns and 'pct' in df_day.columns:
        atrp_condition = (1.5 * df_day['atrp20'] < df_day['pct'].abs())

        for reaction_date in df_day.index[atrp_condition]:
            reaction_idx = df_day.index.get_loc(reaction_date)

            # Boundary Check
            if reaction_idx < OFFSET_DAYS or reaction_idx >= len(df_day) - OFFSET_DAYS:
                continue

            reaction_volume = df_day.iloc[reaction_idx].v
            if reaction_volume < MIN_VOLUME:
                disregarded_count += 1
                continue

            meta = events_map.get(reaction_date, {
                'reaction_idx': reaction_idx,
                'eps': np.nan,
                'eps_est': np.nan,
                'earnings_when': np.nan,
                'event_types': set(),
            })

            meta['reaction_idx'] = reaction_idx
            meta['event_types'].add('atrp_breakout')

            events_map[reaction_date] = meta

        # --- 3. Identify Green Line Breakouts (New ATH after consolidation) ---
        # Definition (confirmed ATH level):
        # - ath_high is the last confirmed ATH level (defined by highs)
        # - We ONLY advance ath_high on a day where the CLOSE finishes above the current ath_high
        # - Consolidation requires >= 5 trading days since the last confirmed ATH advance
        if ('h' in df_day.columns) and ('c' in df_day.columns) and (not df_day.empty):
            ath_high = float(df_day['h'].iloc[0])
            last_ath_idx = 0

            for idx in range(1, len(df_day)):
                current_h = df_day['h'].iloc[idx]
                current_c = df_day['c'].iloc[idx]
                if (not np.isfinite(current_h)) or (not np.isfinite(current_c)):
                    continue

                # Breakout trigger: close must clear the last confirmed ATH-high level
                if float(current_c) <= ath_high:
                    continue

                consolidation_days = idx - last_ath_idx
                reaction_idx = idx

                # Only count as a green line breakout if consolidation was long enough
                if consolidation_days >= 5:
                    if reaction_idx >= OFFSET_DAYS and reaction_idx < len(df_day) - OFFSET_DAYS:
                        reaction_date = df_day.index[reaction_idx]
                        reaction_volume = df_day.iloc[reaction_idx].v

                        if reaction_volume >= MIN_VOLUME:
                            meta = events_map.get(reaction_date, {
                                'reaction_idx': reaction_idx,
                                'eps': np.nan,
                                'eps_est': np.nan,
                                'earnings_when': np.nan,
                                'event_types': set(),
                                'consolidation_days': np.nan,
                            })

                            meta['reaction_idx'] = reaction_idx
                            meta['event_types'].add('green_line_breakout')
                            meta['consolidation_days'] = float(consolidation_days)

                            events_map[reaction_date] = meta
                        else:
                            disregarded_count += 1

                # Confirm/advance ATH ONLY AFTER a close above it.
                # Use today's high as the new ATH level baseline going forward.
                ath_high = max(ath_high, float(current_h))
                last_ath_idx = idx

    # --- 4. Identify BB Lower Touch Events (Mean Reversion to Trend, PM-09) ---
    # Condition: low touches/crosses lower Bollinger band while broader trend is intact.
    # Fires on every qualifying touch (no cooldown) to measure raw success rate.
    if (
        'bb_lower' in df_day.columns
        and 'l' in df_day.columns
        and 'ma50_slope' in df_day.columns
        and 'ma200_dist' in df_day.columns
    ):
        bb_condition = (
            (df_day['l'] <= df_day['bb_lower'])
            & (df_day['ma50_slope'] > 0)
            & (df_day['ma200_dist'] > 0)
        )

        for reaction_date in df_day.index[bb_condition]:
            reaction_idx = df_day.index.get_loc(reaction_date)

            # Boundary Check
            if reaction_idx < OFFSET_DAYS or reaction_idx >= len(df_day) - OFFSET_DAYS:
                continue

            reaction_volume = df_day.iloc[reaction_idx].v
            if reaction_volume < MIN_VOLUME:
                disregarded_count += 1
                continue

            meta = events_map.get(reaction_date, {
                'reaction_idx': reaction_idx,
                'eps': np.nan,
                'eps_est': np.nan,
                'earnings_when': np.nan,
                'event_types': set(),
            })

            meta['reaction_idx'] = reaction_idx
            meta['event_types'].add('bb_lower_touch')

            events_map[reaction_date] = meta

    if not events_map:
        if disregarded_count > 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {i+1:4}/{total_symbols}: {ticker:5} | NO EVENTS ({disregarded_count} disregarded due to volume)")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {i+1:4}/{total_symbols}: {ticker:5} | NO EVENTS")
        continue

    events_data = []
    t_detection = time.time() - t0

    # Time: Processing Loop
    t0 = time.time()
    # --- 3. Unified Processing Loop ---
    for reaction_date in sorted(events_map.keys()):
        meta = events_map[reaction_date]
        reaction_idx = meta['reaction_idx']

        # Slice DataFrames
        df_slice_day = df_day.iloc[reaction_idx - OFFSET_DAYS : reaction_idx + OFFSET_DAYS + 1]
        df_tracking_day = df_slice_day[INDICATORS].copy()

        if len(df_tracking_day) < 2 * OFFSET_DAYS + 1 or df_slice_day.c.isna().all():
            continue

        # Normalized close percentage
        ref_c_day = df_tracking_day['c'].iloc[OFFSET_DAYS-1]
        df_tracking_day['cpct'] = (df_tracking_day['c'] - ref_c_day) / ref_c_day * 100

        # Weekly Logic
        idx_week = df_week.index.get_indexer([reaction_date], method='nearest')[0]
        idx_start_week = max(0, idx_week - OFFSET_WEEKS)
        idx_end_week = min(len(df_week), idx_week + OFFSET_WEEKS + 1)

        effective_week_offset = idx_week - idx_start_week

        df_slice_week = df_week.iloc[idx_start_week : idx_end_week]
        df_tracking_week = df_slice_week[INDICATORS].copy()

        if not df_tracking_week.empty:
            ref_c_week = df_tracking_week['c'].iloc[effective_week_offset-1] if effective_week_offset > 0 else df_tracking_week['c'].iloc[0]
            df_tracking_week['cpct'] = (df_tracking_week['c'] - ref_c_week) / ref_c_week * 100
            df_tracking_week.columns = [f'w_{col}' for col in df_tracking_week.columns]

        # SPY Comparison Logic
        if not any(df_spy_day.index == reaction_date):
            continue

        idx_spy = df_spy_day.index.get_loc(reaction_date)
        spy_ref_val = df_spy_day['c'].iloc[idx_spy - OFFSET_DAYS]
        spy_slice = (df_spy_day['c'].iloc[idx_spy - OFFSET_DAYS : idx_spy + OFFSET_DAYS + 1] - spy_ref_val) / spy_ref_val * 100
        df_tracking_day['spy'] = spy_slice.values if len(spy_slice) == len(df_tracking_day) else np.nan

        # SPY Indicators
        spy_slice_ind = df_spy_day.iloc[idx_spy - OFFSET_DAYS : idx_spy + OFFSET_DAYS + 1]
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

        if len(spy_slice_week) == len(df_tracking_week):
             df_tracking_week['w_spy'] = spy_slice_week.values
        else:
             df_tracking_week['w_spy'] = np.nan

        # --- Flattening ---
        df_tracking_day.index = np.arange(len(df_tracking_day)) - OFFSET_DAYS
        df_tracking_week.index = np.arange(len(df_tracking_week)) - effective_week_offset

        flat_day = {f"{col}{idx}": val for (col, idx), val in df_tracking_day.unstack().items()}
        flat_week = {f"{col}{idx}": val for (col, idx), val in df_tracking_week.unstack().items()}

        event_types = sorted(list(meta.get('event_types', set())))
        is_earnings = ('earnings' in event_types)
        event_type = "|".join(event_types) if event_types else "unknown"

        event_row = {
            'symbol': ticker,
            'date': reaction_date,
            'datasource': swing_data_full.datasource,

            # Keep both if you want
            'event_types': event_types,

            # Existing
            'is_earnings': is_earnings,

            # NEW: flat flags for fast filtering
            'evt_atrp_breakout': ('atrp_breakout' in event_types),
            'evt_green_line_breakout': ('green_line_breakout' in event_types),
            'evt_bb_lower_touch': ('bb_lower_touch' in event_types),

            # ... existing fields ...
            'is_etf': is_etf,
            'eps': meta.get('eps', np.nan),
            'eps_est': meta.get('eps_est', np.nan),
            'earnings_when': meta.get('earnings_when', np.nan),
            'gappct': df_day.iloc[reaction_idx].gappct,
            'market_cap': np.nan,
            'market_cap_date': pd.NaT,
            'market_cap_class': 'Unknown',
            'original_price': df_day.iloc[reaction_idx].original_price if 'original_price' in df_day.columns else np.nan,

            # Performance: use values computed in indicators.py (df_day) instead of recomputing here
            '1M_chg': df_day.iloc[reaction_idx]['1M_chg'] if '1M_chg' in df_day.columns else np.nan,
            '3M_chg': df_day.iloc[reaction_idx]['3M_chg'] if '3M_chg' in df_day.columns else np.nan,
            '6M_chg': df_day.iloc[reaction_idx]['6M_chg'] if '6M_chg' in df_day.columns else np.nan,
            '12M_chg': df_day.iloc[reaction_idx]['12M_chg'] if '12M_chg' in df_day.columns else np.nan,
        }

        # Market Cap Lookup
        if ts_market_cap is not None and not ts_market_cap.empty:
            mkp_idx = ts_market_cap.index.get_indexer([reaction_date], method="nearest")[0]
            mc_row = ts_market_cap.iloc[mkp_idx]

            mcap_date = mc_row.name
            if abs((reaction_date - mcap_date).days) <= 730:
                event_row['market_cap'] = mc_row['market_cap']
                event_row['market_cap_date'] = mcap_date
                event_row['market_cap_class'] = mc_row['market_cap_class']
            else:
                event_row['market_cap_class'] = 'Unknown'

        full_row = {**event_row, **flat_day, **flat_week}
        events_data.append(full_row)

    if not events_data:
        # This case should be rare if events_map had items, but possible due to inner loop skips
        disregarded_str = f" ({disregarded_count} disregarded due to volume)" if disregarded_count > 0 else ""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {i+1:4}/{total_symbols}: {ticker:5} | NO EVENTS{disregarded_str}")
        continue

    df_ticker_events = pd.DataFrame(events_data)

    # Save Parquet Data
    ticker_file = f'{data_path}/{ticker}.parquet'
    df_ticker_events.to_parquet(ticker_file, index=False)

    total_time = time.time() - ticker_start
    evt_types = df_ticker_events['event_types'].explode().value_counts().to_dict()
    evt_str = ", ".join([f"{k}: {v}" for k, v in evt_types.items()])
    disregarded_str = f" | Disregarded: {disregarded_count}" if disregarded_count > 0 else ""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {i+1:4}/{total_symbols}: {ticker:5} | Found {len(df_ticker_events)} events ({evt_str}){disregarded_str} | Total: {total_time:.2f}s")

#%%
# --- 4. Final Aggregation ---
print("Aggregating all files...")
all_data = []
all_data_filename = f'finance/_data/{output_name}/all.parquet'
for ticker in liquid_symbols:
    filename = f'{data_path}/{ticker}.parquet'
    if os.path.exists(filename):
        all_data.append(pd.read_parquet(filename))

#%%
if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    df_all.to_parquet(all_data_filename, index=False)
    print(f"Complete. Saved to {all_data_filename}")

#%% Split file into chunks by year starting from < 2010, 2011, ... 2025
previous_year = 1900
for year in range(2010, 2026):
    filename = f'finance/_data/{output_name}/all_{year}.parquet'
    df_year = df_all[(df_all.date.dt.year > previous_year) & (df_all.date.dt.year <= year)]
    df_year.to_parquet(filename, index=False)
    print(f"Complete. Saved to {filename}")
    previous_year = year
