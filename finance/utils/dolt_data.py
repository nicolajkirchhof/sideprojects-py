import os.path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from finance import utils

#%%
# MySQL connection setup (localhost:3306)
DB_URL = 'mysql+pymysql://root:@localhost:3306/'
db_stocks_connection = create_engine(DB_URL + 'stocks')
db_options_connection = create_engine(DB_URL + 'options')
db_earnings_connection = create_engine(DB_URL + 'earnings')

def daily_time_range(symbol, start_date=None, end_date=None):
  ohclv_query = """select * from ohlcv where act_symbol = :symbol and date >= :start_date and date <= :end_date"""
  # Example usage with pandas:
  df_stk = pd.read_sql(text(ohclv_query), db_stocks_connection, params={'symbol': symbol, 'start_date': start_date, 'end_date': end_date})

  df_stk = df_stk.rename(columns={'act_symbol':'symbol', 'open':'o', 'close':'c', 'high':'h', 'low':'l', 'volume':'v'})
  df_stk['date'] = pd.to_datetime(df_stk.date)
  df_stk.set_index('date', inplace=True)

def symbol_info(symbol):
  query = """select * from symbol where act_symbol = :symbol"""
  df = pd.read_sql(text(query), db_stocks_connection, params={'symbol': symbol})
  df = df.rename(columns={'act_symbol':'symbol', 'security_name': 'name'})
  if df.empty: return None
  return df.iloc[0]


def financial_info(symbol):
  if os.path.exists(f'finance/_data/financials/{symbol}.csv'):
    df_financial = pd.read_csv(f'finance/_data/financials/{symbol}.csv', index_col='date', parse_dates=True)
    return df_financial
  query = """select date, shares_outstanding from balance_sheet_equity where act_symbol = :symbol"""
  df = pd.read_sql(text(query), db_earnings_connection, params={'symbol': symbol})
  df['date'] = pd.to_datetime(df.date)
  df.set_index('date', inplace=True)
  return df


#%%
def daily_w_volatility(symbol):
  #%%
  ohclv_query = """select * from ohlcv where act_symbol = :symbol"""
  # Example usage with pandas:
  df_stk = pd.read_sql(text(ohclv_query), db_stocks_connection, params={'symbol': symbol})

  df_stk = df_stk.rename(columns={'act_symbol':'symbol', 'open':'o', 'close':'c', 'high':'h', 'low':'l', 'volume':'v'})
  df_stk['date'] = pd.to_datetime(df_stk.date)
  df_stk.set_index('date', inplace=True)
  df_stk[utils.definitions.OHLCLV] = df_stk[utils.definitions.OHLCLV].interpolate()

  volatility_query = """select act_symbol, date, hv_current, iv_current, iv_year_high, iv_year_low from volatility_history where act_symbol = :symbol"""
  df_vol = pd.read_sql(text(volatility_query), db_options_connection, params={'symbol': symbol})
  if not df_vol.empty:
    df_vol = df_vol.rename(columns={'act_symbol':'symbol', 'iv_current':'iv', 'hv_current':'hv'})
    df_vol['date'] = pd.to_datetime(df_vol.date)
    df_vol.set_index('date', inplace=True)
    cols_to_fix = ['iv', 'hv', 'iv_year_high', 'iv_year_low']
    df_vol = df_vol.sort_values('date')

    #%%
    window = 252
    # 3. IV Rank: (Current - Min) / (Max - Min) * 100
    df_vol['iv_rank'] = ((df_vol['iv'] - df_vol.iv_year_high) / (df_vol.iv_year_high - df_vol.iv_year_low)) * 100

    # 4. IV Percentile: Percentage of days in window where IV was lower than current IV
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

  df_stk_vol[utils.definitions.OHLCLV+utils.definitions.IV] = df_stk_vol[utils.definitions.OHLCLV+utils.definitions.IV].interpolate()
  df_stk_vol['symbol'] = symbol
  #%%
  return df_stk_vol



