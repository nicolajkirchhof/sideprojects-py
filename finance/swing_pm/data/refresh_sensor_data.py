'''
This script refreshes sensor data for the Swing PM project.
It fetches data from the sensor API and updates the local database.
'''
import pandas as pd
from time import perf_counter
from datetime import datetime

from sqlalchemy import text

from finance import utils

# %load_ext autoreload
# %autoreload 2
# Restart issue BVN

#%%

t0_all = perf_counter()
print(f"======>>>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] refresh_sensor_data: start")

query_delisted = """
  select * from symbol
  where last_seen < date_sub((select max(last_seen) from symbol), interval 3 day)
"""
df_delisted = pd.read_sql(text(query_delisted), utils.dolt_data.db_stocks_connection)
df_delisted.to_csv('finance/_data/state/delisted.csv', index=False)
delisted = set(df_delisted['act_symbol'].sort_values().tolist())
pd.DataFrame({'symbol': sorted(delisted)}).to_parquet('finance/_data/state/delisted.parquet', index=False)
print(f"======>>>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] delisted: {len(df_delisted):,} rows (from dolt stocks.symbol)")

#%%
underlyings = utils.underlyings.get_liquid_underlyings()
total = len(underlyings)
print(f"======>>>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] underlyings: {total:,}")

ok = 0
errors = 0
ib_con = utils.ibkr.connect('api_paper', 17, 1)
delisted_ibkr = []
non_delisted_dolt = []

for i, symbol in enumerate(underlyings, start=1):
  t0 = perf_counter()
  df_day = None
  try:
    df_day = utils.ibkr.daily_w_volatility(symbol, offline=False, ib_con=ib_con)
    is_delisted = symbol in delisted
    # df_day = utils.ibkr.daily_w_volatility(symbol, offline=True)

    if is_delisted and df_day is not None:
      print(f"======>>> {symbol} is delisted but available IBKR cache")
      delisted_ibkr.append(symbol)
    if not is_delisted and df_day is None:
      print(f"======>>> {symbol} is NOT delisted but NOT in IBKR cache")
      non_delisted_dolt.append(symbol)

    if df_day is None:
      utils.dolt_data.daily_w_volatility(symbol)
    # utils.SwingTradingData(symbol, datasource='update')
    utils.dolt_data.splits(symbol, offline=False)
    utils.dolt_data.financial_info(symbol, offline=False)
    ok += 1
  except Exception as e:
    errors += 1
    print(f"======>>>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{i}/{total}] {symbol}: ERROR {type(e).__name__}: {e}")
    continue

  dt = perf_counter() - t0

  elapsed = perf_counter() - t0_all
  rate = i / elapsed if elapsed > 0 else 0.0
  eta_s = (total - i) / rate if rate > 0 else float("inf")
  print(f"======>>>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{i}/{total}] {symbol}: ok {dt:.2f}s | ok={ok} err={errors} ETA={eta_s/60:.1f}m")

elapsed_all = perf_counter() - t0_all
print(f"======>>>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] refresh_sensor_data: done in {elapsed_all/60:.2f}m | ok={ok} err={errors}")
print(f"======>>>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] delisted_ibkr: {delisted_ibkr}")
print(f"======>>>[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] non_delisted_dolt: {non_delisted_dolt}")
