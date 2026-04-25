"""One-time export of all intraday data from InfluxDB to per-symbol Parquet files.

Usage:
  uv run python finance/ibkr/data/export_influxdb.py
  uv run python finance/ibkr/data/export_influxdb.py --schema index
  uv run python finance/ibkr/data/export_influxdb.py --schema index --symbol SPX
"""
import argparse
from datetime import datetime

import influxdb as idb
import pandas as pd

from finance.utils.intraday import symbol_path, BAR_COLUMNS

INFLUX_HOST = 'localhost'
INFLUX_PORT = 8086

# InfluxDB databases -> Parquet schema folders
DB_TO_SCHEMA = {
  'index': 'index',
  'cfd': 'cfd',
  'forex': 'forex',
  'stk': 'stk',
  'future': 'future',
}

CHUNK_SIZE = 500_000


def get_measurements(client: idb.InfluxDBClient, database: str) -> list[str]:
  result = client.query('SHOW MEASUREMENTS', database=database)
  return [row['name'] for row in result.get_points()]


def export_measurement(client_df: idb.DataFrameClient, database: str,
                       measurement: str, schema: str) -> int:
  """Export one measurement to a Parquet file. Returns row count."""
  path = symbol_path(measurement, schema)
  path.parent.mkdir(parents=True, exist_ok=True)

  # Paginated read to handle large measurements
  offset = 0
  chunks = []
  while True:
    query = f'SELECT * FROM "{measurement}" LIMIT {CHUNK_SIZE} OFFSET {offset}'
    result = client_df.query(query, database=database)
    if measurement not in result or result[measurement].empty:
      break
    chunks.append(result[measurement])
    fetched = len(result[measurement])
    offset += fetched
    if fetched < CHUNK_SIZE:
      break

  if not chunks:
    print(f'  {measurement}: EMPTY — skipped')
    return 0

  df = pd.concat(chunks)
  df.index.name = 'time'

  # Keep only known bar columns that exist in this measurement
  cols = [c for c in BAR_COLUMNS if c in df.columns]
  df = df[cols]

  # Drop rows that are all-NaN (can happen with sparse IV/HV data)
  ohlcv_cols = [c for c in ['o', 'h', 'l', 'c'] if c in df.columns]
  df = df.dropna(subset=ohlcv_cols, how='all')

  df.sort_index(inplace=True)
  df = df[~df.index.duplicated(keep='last')]
  df.to_parquet(path)

  return len(df)


def main():
  parser = argparse.ArgumentParser(description='Export InfluxDB intraday data to Parquet')
  parser.add_argument('--schema', choices=list(DB_TO_SCHEMA.values()),
                      help='Export only this schema')
  parser.add_argument('--symbol', help='Export only this symbol (requires --schema)')
  args = parser.parse_args()

  if args.symbol and not args.schema:
    parser.error('--symbol requires --schema')

  client = idb.InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT, timeout=300)
  client_df = idb.DataFrameClient(host=INFLUX_HOST, port=INFLUX_PORT, timeout=300)

  total_rows = 0
  total_symbols = 0

  dbs_to_export = {args.schema: args.schema} if args.schema else DB_TO_SCHEMA

  for db, schema in dbs_to_export.items():
    print(f'\n=== {db} -> {schema}/ ===')
    measurements = get_measurements(client, db)

    if args.symbol:
      measurements = [m for m in measurements if m == args.symbol]
      if not measurements:
        print(f'  Symbol {args.symbol} not found in {db}')
        continue

    for measurement in measurements:
      rows = export_measurement(client_df, db, measurement, schema)
      if rows > 0:
        total_rows += rows
        total_symbols += 1
        print(f'  {measurement}: {rows:,} rows')

  print(f'\nDone: {total_symbols} symbols, {total_rows:,} total rows')


if __name__ == '__main__':
  main()
