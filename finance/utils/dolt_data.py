import os.path
import pickle
import time
import functools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from finance import utils


DELISTED = None

#%%
# MySQL connection setup (localhost:3306)
DB_URL = 'mysql+pymysql://root:@localhost:3306/'
db_stocks_connection = None
db_options_connection = None
db_earnings_connection = None


def init_engines():
  global db_stocks_connection, db_options_connection, db_earnings_connection
  db_options_connection = create_engine(DB_URL + 'options')
  db_earnings_connection = create_engine(DB_URL + 'earnings')
  db_stocks_connection = create_engine(DB_URL + 'stocks')

init_engines()

# ---- Local PKL cache (fast path) ----
_CACHE_DIR = Path("finance/_data/dolt_cache")

def _cache_file(func_name: str, symbol: str) -> Path:
  _CACHE_DIR.mkdir(parents=True, exist_ok=True)
  safe_symbol = str(symbol).upper().strip()
  return _CACHE_DIR / f"{func_name}__{safe_symbol}.pkl"

def _cache_is_fresh(path: Path, *, max_age_days: float | None) -> bool:
  if not path.exists():
    return False
  if max_age_days is None:
    return True
  age_seconds = time.time() - path.stat().st_mtime
  return age_seconds <= max_age_days * 86400.0

def _read_cache(path: Path) -> pd.DataFrame | None:
  try:
    obj = pd.read_pickle(path)
    return obj if isinstance(obj, pd.DataFrame) else None
  except Exception:
    return None

def _write_cache(path: Path, df: pd.DataFrame) -> None:
  tmp = path.with_suffix(path.suffix + ".tmp")
  df.to_pickle(tmp)
  tmp.replace(path)

def time_db_call(func):
    """Decorator to time database function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"  [DB] {func.__name__} took {duration:.4f}s")
        return result
    return wrapper

@time_db_call
def daily_time_range(symbol, start_date=None, end_date=None):
  ohclv_query = """select * from ohlcv where act_symbol = :symbol and date >= :start_date and date <= :end_date"""
  df_stk = pd.read_sql(
    text(ohclv_query),
    db_stocks_connection,
    params={'symbol': symbol, 'start_date': start_date, 'end_date': end_date}
  )

  df_stk = df_stk.rename(columns={'act_symbol':'symbol', 'open':'o', 'close':'c', 'high':'h', 'low':'l', 'volume':'v'})
  df_stk['date'] = pd.to_datetime(df_stk.date)
  df_stk.set_index('date', inplace=True)
  return df_stk

@time_db_call
def symbol_info(symbol):
  query = """select * from symbol where act_symbol = :symbol"""
  df = pd.read_sql(text(query), db_stocks_connection, params={'symbol': symbol})
  df = df.rename(columns={'act_symbol':'symbol', 'security_name': 'name'})
  if df.empty: return None
  return df.iloc[0]

@time_db_call
def financial_info(symbol, offline=False):
  cache_path = _cache_file("financial_info", symbol)
  if offline:
    cached = _read_cache(cache_path)
    return cached if cached is not None else pd.DataFrame()

  query = """select date, shares_outstanding from balance_sheet_equity where act_symbol = :symbol"""
  df = pd.read_sql(text(query), db_earnings_connection, params={'symbol': symbol})
  df['date'] = pd.to_datetime(df.date)
  df.set_index('date', inplace=True)
  
  if not df.empty:
    _write_cache(cache_path, df)
    
  return df

@time_db_call
def splits(symbol, offline=False):
  cache_path = _cache_file("splits", symbol)
  if offline:
    cached = _read_cache(cache_path)
    return cached if cached is not None else pd.DataFrame()

  query = """select act_symbol, ex_date, to_factor, for_factor from split where act_symbol = :symbol"""
  df = pd.read_sql(text(query), db_stocks_connection, params={'symbol': symbol})
  df.rename(columns={'act_symbol':'symbol', 'ex_date': 'date'}, inplace=True)
  df['date'] = pd.to_datetime(df.date)
  df.set_index('date', inplace=True)

  if not df.empty:
    _write_cache(cache_path, df)

  return df

#%%
def load_delisted():
  global DELISTED
  if DELISTED is not None: return DELISTED

  if os.path.exists('finance/_data/delisted.pkl'):
    DELISTED = pickle.load(open('finance/_data/delisted.pkl', 'rb'))
  else:
    print("No delisted.pkl found. DOLT will not work correctly")
  return DELISTED
#%%
@time_db_call
def daily_w_volatility(
    symbol: str,
    offline: bool = False,
) -> pd.DataFrame:
  """
  Load daily OHLCV and merge volatility history, with optional PKL caching.

  Parameters
  ----------
  max_age_days:
      Cache freshness window. None = never expires.
  offline:
      If True, only load from cache, do not hit DB.
  """
  delisted = load_delisted()
  cache_path = _cache_file("daily_w_volatility", symbol)
  cached = _read_cache(cache_path)

  if offline or (symbol in delisted and cached is not None):
    print(f"{symbol} is delisted, returning cached data.")
    return cached

  ohclv_query = """select * from ohlcv where act_symbol = :symbol"""
  df_stk = pd.read_sql(text(ohclv_query), db_stocks_connection, params={'symbol': symbol})

  df_stk = df_stk.rename(columns={'act_symbol':'symbol', 'open':'o', 'close':'c', 'high':'h', 'low':'l', 'volume':'v'})
  df_stk['date'] = pd.to_datetime(df_stk.date)
  df_stk.set_index('date', inplace=True)
  df_stk[utils.definitions.OHLCLV] = df_stk[utils.definitions.OHLCLV].interpolate()

  volatility_query = """select act_symbol, date, hv_current, iv_current, iv_year_high, iv_year_low from volatility_history where act_symbol = :symbol"""
  df_vol = pd.read_sql(text(volatility_query), db_options_connection, params={'symbol': symbol})
  df_vol.dropna(subset=['iv_current'], inplace=True)
  if not df_vol.empty:
    df_vol = df_vol.rename(columns={'act_symbol':'symbol', 'iv_current':'iv', 'hv_current':'hv'})
    df_vol['date'] = pd.to_datetime(df_vol.date)
    df_vol.set_index('date', inplace=True)
    df_vol = df_vol.sort_values('date')

    window = 252
    df_vol['iv_rank'] = ((df_vol['iv'] - df_vol.iv_year_high) / (df_vol.iv_year_high - df_vol.iv_year_low)) * 100

    def calculate_percentile(x):
      if len(x) < window: return None
      current = x.iloc[-1]
      return (x < current).sum() / len(x) * 100

    df_vol['iv_pct'] = (
      df_vol['iv']
      .rolling(window=window)
      .apply(calculate_percentile, raw=False)
    )

    df_stk_vol = pd.merge(df_stk[utils.definitions.OHLCLV], df_vol[utils.definitions.IV], on='date', how='outer')
  else:
    df_stk_vol = df_stk
    df_stk_vol[utils.definitions.IV] = np.nan

  df_stk_vol[utils.definitions.OHLCLV + utils.definitions.IV] = df_stk_vol[utils.definitions.OHLCLV + utils.definitions.IV].interpolate()
  df_stk_vol['symbol'] = symbol

  _write_cache(cache_path, df_stk_vol)

  return df_stk_vol



