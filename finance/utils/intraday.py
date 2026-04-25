"""Read and write intraday OHLCV bars stored as per-symbol Parquet files.

Layout:
  finance/_data/intraday/{schema}/{SYMBOL}.parquet

Each file has a UTC DatetimeIndex named 'time' and columns matching BAR_COLUMNS.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / '_data' / 'intraday'

SCHEMAS = ['index', 'cfd', 'forex', 'stk', 'future']

BAR_COLUMNS = ['o', 'h', 'l', 'c', 'v', 'a', 'bc',
               'hvo', 'hvh', 'hvl', 'hvc',
               'ivo', 'ivh', 'ivl', 'ivc']

SEC_TYPE_SCHEMA_MAPPING = {
  'IND': 'index',
  'CFD': 'cfd',
  'CONTFUT': 'future',
  'FUT': 'future',
  'CASH': 'forex',
  'STK': 'stk',
  'CMDTY': 'cfd',
}

OHLCV_AGG = {
  'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last',
  'v': 'sum', 'a': 'last', 'bc': 'sum',
  'hvo': 'first', 'hvh': 'max', 'hvl': 'min', 'hvc': 'last',
  'ivo': 'first', 'ivh': 'max', 'ivl': 'min', 'ivc': 'last',
}


def symbol_path(symbol: str, schema: str, *, data_dir: Path | None = None) -> Path:
  base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
  return base / schema / f'{symbol}.parquet'


def get_bars(symbol: str, schema: str, start: datetime, end: datetime,
             period: str | None = None, *, data_dir: Path | None = None) -> pd.DataFrame:
  """Read OHLCV bars for a symbol, optionally resampled.

  Parameters
  ----------
  period : pandas frequency string, e.g. '5min', '1h', '1D'.
           When None, returns raw 1-min bars.
  """
  path = symbol_path(symbol, schema, data_dir=data_dir)
  if not path.exists():
    return pd.DataFrame()

  df = pd.read_parquet(path)
  df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]

  if period is not None and not df.empty:
    agg = {col: OHLCV_AGG[col] for col in df.columns if col in OHLCV_AGG}
    df = df.resample(period).agg(agg).dropna(subset=['o'])

  return df


def get_last(symbol: str, schema: str, *, data_dir: Path | None = None) -> pd.Timestamp | None:
  """Return the last timestamp for a symbol, or None if no data."""
  path = symbol_path(symbol, schema, data_dir=data_dir)
  if not path.exists():
    return None
  # Read only the first data column to keep the index; pandas drops the index
  # when columns=[] is used with a DatetimeIndex stored in Parquet metadata.
  import pyarrow.parquet as pq
  schema = pq.read_schema(path)
  first_col = schema.names[0]
  df = pd.read_parquet(path, columns=[first_col])
  if df.empty:
    return None
  return df.index.max()


def list_symbols(schema: str, *, data_dir: Path | None = None) -> list[str]:
  """Return sorted list of symbols available in a schema."""
  base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
  schema_dir = base / schema
  if not schema_dir.exists():
    return []
  return sorted(p.stem for p in schema_dir.glob('*.parquet'))


def write_bars(df: pd.DataFrame, symbol: str, schema: str,
               *, data_dir: Path | None = None) -> int:
  """Write (or append + deduplicate) bars to a symbol's Parquet file.

  Expects a DataFrame with DatetimeIndex named 'time' and bar columns.
  On overlap, newer rows (from df) win.
  Returns the total number of rows in the written file.
  """
  if df.empty:
    return 0

  path = symbol_path(symbol, schema, data_dir=data_dir)
  path.parent.mkdir(parents=True, exist_ok=True)

  if path.exists():
    existing = pd.read_parquet(path)
    combined = pd.concat([existing, df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)
  else:
    combined = df.sort_index()

  combined.index.name = 'time'

  tmp_path = path.with_suffix('.parquet.tmp')
  combined.to_parquet(tmp_path)
  os.replace(tmp_path, path)

  return len(combined)


def sec_type_to_schema(sec_type: str) -> str:
  return SEC_TYPE_SCHEMA_MAPPING[sec_type]
