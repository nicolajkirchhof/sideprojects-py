# %%
import os
import time
from datetime import datetime

import pandas as pd
import matplotlib as mpl
from sqlalchemy import create_engine, text

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%

# If True, reprocess and overwrite existing earnings_cleaned/*.csv
FORCE = True

df_ecsv = pd.read_csv('finance/_data/hist_earnings/earnings_latest.csv', sep=',', parse_dates=['date'])
df_ecsv = df_ecsv.rename(
  columns={'symbol': 'symbol_csv', 'eps_est': 'eps_est_csv', 'eps': 'eps_csv', 'qtr': 'qtr_csv', 'date': 'date_csv',
           'release_time': 'when_csv'})

# %%
db_connection_str = 'mysql+pymysql://root:@localhost:3306/earnings'
db_connection = create_engine(db_connection_str)

liquid_stocks = pd.read_pickle('finance/_data/liquid_stocks.pkl')

out_dir = 'finance/_data/earnings_cleaned'
os.makedirs(out_dir, exist_ok=True)

total = len(liquid_stocks)
t0_all = time.time()

written = 0
skipped_existing = 0
empty_join = 0
errors = 0

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting earnings merge for {total} tickers... FORCE={FORCE}")

for i, ticker in enumerate(liquid_stocks, start=1):
  t0 = time.time()
  out_path = f'{out_dir}/{ticker}.csv'

  if (not FORCE) and os.path.exists(out_path):
    skipped_existing += 1
    if i == 1 or i % 250 == 0 or i == total:
      elapsed = time.time() - t0_all
      rate = i / elapsed if elapsed > 0 else 0.0
      eta_s = (total - i) / rate if rate > 0 else float("inf")
      print(
        f"[{i}/{total}] {ticker}: exists -> skip | written={written} empty={empty_join} errors={errors} ETA={eta_s / 60:.1f}m")
    continue

  try:
    query = """select ec.act_symbol, ec.date, ec.when, eh.period_end_date, eh.reported, eh.estimate
               from earnings_calendar ec
                        left join eps_history eh on ec.act_symbol = eh.act_symbol
                   and eh.period_end_date = (select max(period_end_date)
                                             from eps_history eh_sub
                                             where eh_sub.act_symbol = ec.act_symbol
                                               and eh_sub.period_end_date < ec.date)
               where ec.act_symbol = :ticker"""
    stmt = text(query)

    df_edb = pd.read_sql(stmt, db_connection, params={'ticker': ticker})

    df_edb = df_edb[['act_symbol', 'date', 'when', 'period_end_date', 'reported', 'estimate']].rename(
      columns={'act_symbol': 'symbol', 'reported': 'eps', 'estimate': 'eps_est'}
    )
    df_edb['date'] = pd.to_datetime(df_edb.date)
    df_edb['when'] = df_edb.when.str.replace('After market close', 'post', regex=False)
    df_edb['when'] = df_edb.when.str.replace('Before market open', 'pre', regex=False)

    df_escv_ticker = df_ecsv[df_ecsv.symbol_csv == ticker]
    df_join = df_edb.merge(df_escv_ticker, left_on='date', right_on='date_csv', how='outer')

    for key in ['symbol', 'date', 'when', 'eps_est', 'eps']:
      df_join[key] = df_join[key].combine_first(df_join.get(f'{key}_csv'))

    if df_join.empty:
      empty_join += 1
      if i == 1 or i % 250 == 0 or i == total:
        elapsed = time.time() - t0_all
        rate = i / elapsed if elapsed > 0 else 0.0
        eta_s = (total - i) / rate if rate > 0 else float("inf")
        print(
          f"[{i}/{total}] {ticker}: empty -> skip | written={written} empty={empty_join} errors={errors} ETA={eta_s / 60:.1f}m")
      continue

    df_join.to_csv(out_path, index=False)
    written += 1

    if i <= 25 or i % 100 == 0 or i == total:
      elapsed = time.time() - t0_all
      rate = i / elapsed if elapsed > 0 else 0.0
      eta_s = (total - i) / rate if rate > 0 else float("inf")
      print(
        f"[{i}/{total}] {ticker}: wrote {len(df_join):,} rows in {(time.time() - t0):.2f}s "
        f"| written={written} skip={skipped_existing} empty={empty_join} errors={errors} ETA={eta_s / 60:.1f}m"
      )

  except Exception as e:
    errors += 1
    print(f"[{i}/{total}] {ticker}: ERROR {type(e).__name__}: {e}")

# %%
elapsed_all = time.time() - t0_all
print(
  f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done in {elapsed_all / 60:.2f} min | "
  f"written={written} skipped_existing={skipped_existing} empty={empty_join} errors={errors}"
)
# %%
