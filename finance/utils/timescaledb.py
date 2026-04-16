from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = 'postgresql+psycopg://postgres@localhost:5454/intraday'
engine = create_engine(DB_URL)

# Schema names matching old InfluxDB database constants
SCHEMA_INDEX = 'index'
SCHEMA_CFD = 'cfd'
SCHEMA_FOREX = 'forex'
SCHEMA_STK = 'stk'
SCHEMA_FUTURE = 'future'
SCHEMA_DAILY = 'daily'
SCHEMA_SWING = 'swing'

SEC_TYPE_SCHEMA_MAPPING = {
  'IND': SCHEMA_INDEX,
  'CFD': SCHEMA_CFD,
  'CONTFUT': SCHEMA_FUTURE,
  'FUT': SCHEMA_FUTURE,
  'CASH': SCHEMA_FOREX,
  'STK': SCHEMA_STK,
  'CMDTY': SCHEMA_CFD,
}

# All bar columns (excluding time and symbol)
BAR_COLUMNS = ['o', 'h', 'l', 'c', 'v', 'a', 'bc',
               'hvo', 'hvh', 'hvl', 'hvc',
               'ivo', 'ivh', 'ivl', 'ivc']


def get_bars(symbol: str, schema: str, start: datetime, end: datetime,
             period: str | None = None) -> pd.DataFrame:
  """Query OHLCV bars, optionally aggregated by time_bucket.

  Parameters
  ----------
  period : optional bucket width, e.g. '1 hour', '1 day', '5 min'.
           When None, returns raw 1-min bars.
  """
  if period is None:
    query = text("""
      SELECT time, o, h, l, c, v, a, bc, hvo, hvh, hvl, hvc, ivo, ivh, ivl, ivc
      FROM :schema.bars
      WHERE symbol = :symbol AND time >= :start AND time < :end
      ORDER BY time
    """.replace(':schema', f'"{schema}"'))
    params = {'symbol': symbol, 'start': start, 'end': end}
  else:
    query = text("""
      SELECT time_bucket(:period, time) AS time,
             first(o, time) AS o, max(h) AS h, min(l) AS l, last(c, time) AS c,
             sum(v) AS v, last(a, time) AS a, sum(bc) AS bc,
             first(hvo, time) AS hvo, max(hvh) AS hvh, min(hvl) AS hvl, last(hvc, time) AS hvc,
             first(ivo, time) AS ivo, max(ivh) AS ivh, min(ivl) AS ivl, last(ivc, time) AS ivc
      FROM :schema.bars
      WHERE symbol = :symbol AND time >= :start AND time < :end
      GROUP BY 1 ORDER BY 1
    """.replace(':schema', f'"{schema}"'))
    params = {'symbol': symbol, 'start': start, 'end': end, 'period': period}

  df = pd.read_sql(query, engine, params=params, parse_dates=['time'])
  if not df.empty:
    df.set_index('time', inplace=True)
  return df


def get_last(symbol: str, schema: str) -> datetime | None:
  """Return the last timestamp for a symbol, or None if no data."""
  query = text("""
    SELECT max(time) AS last_time
    FROM :schema.bars
    WHERE symbol = :symbol
  """.replace(':schema', f'"{schema}"'))
  with engine.connect() as conn:
    result = conn.execute(query, {'symbol': symbol}).scalar()
  return result


def write_bars(df: pd.DataFrame, symbol: str, schema: str) -> int:
  """Write a DataFrame of bars to TimescaleDB.

  Expects columns matching the InfluxDB field names (o, h, l, c, v, a, bc,
  and optionally hvo/hvh/hvl/hvc, ivo/ivh/ivl/ivc).
  Index must be a DatetimeIndex (UTC).

  Returns the number of rows written.
  """
  if df.empty:
    return 0

  df_write = df.copy()
  df_write['symbol'] = symbol
  df_write.index.name = 'time'
  df_write = df_write.reset_index()

  # Keep only known columns
  cols = ['time', 'symbol'] + [c for c in BAR_COLUMNS if c in df_write.columns]
  df_write = df_write[cols]

  with engine.begin() as conn:
    # Upsert: insert rows, on conflict update all bar columns
    bar_cols_present = [c for c in BAR_COLUMNS if c in df_write.columns]
    col_list = ', '.join(['time', 'symbol'] + bar_cols_present)
    placeholders = ', '.join([f':{c}' for c in ['time', 'symbol'] + bar_cols_present])
    update_set = ', '.join([f'{c} = EXCLUDED.{c}' for c in bar_cols_present])

    upsert_sql = text(f"""
      INSERT INTO "{schema}".bars ({col_list})
      VALUES ({placeholders})
      ON CONFLICT (time, symbol) DO UPDATE SET {update_set}
    """)

    conn.execute(upsert_sql, df_write.to_dict('records'))

  return len(df_write)


def sec_type_to_schema(sec_type: str) -> str:
  return SEC_TYPE_SCHEMA_MAPPING[sec_type]
