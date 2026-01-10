import os
from typing import Literal

import dateutil
import ib_async as ib
import numpy as np
from datetime import datetime, timedelta

import pandas as pd

from finance import utils

AVAILABLE_TYPES_OF_DATA = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY',
                           'OPTION_IMPLIED_VOLATILITY']
TYPES_OF_DATA = {'IND': ['TRADES', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY'], 'CFD': ['MIDPOINT'],
                 'FUT': ['TRADES'],
                 'STK': ['TRADES', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY'], 'CONTFUT': ['TRADES'],
                 'CASH': ['MIDPOINT'], 'CMDTY': ['MIDPOINT']}

FIELD_NAME_LU = {
  'TRADES': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'MIDPOINT': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'HISTORICAL_VOLATILITY': {'open': 'hvo', 'high': 'hvh', 'low': 'hvl', 'close': 'hvc', 'volume': 'hvv',
                            'average': 'hva', 'barCount': 'hvbc'},
  'OPTION_IMPLIED_VOLATILITY': {'open': 'ivo', 'high': 'ivh', 'low': 'ivl', 'close': 'ivc', 'volume': 'ivv',
                                'average': 'iva', 'barCount': 'ivbc'},
}

#%%
def get_options_price(market_data, type):
  return getattr(market_data, type) if getattr(market_data, type) > 0 else getattr(market_data, 'prev'+type.capitalize())

def connect(instance, id, data_type):
  ib.util.startLoop()
  ib_con = ib.IB()
  tws_ports = {'real': 7497, 'paper': 7498, 'api': 4001, 'api_paper': 4002}
  ib_con.connect('127.0.0.1', tws_ports[instance], clientId=id, readonly=True)
  ib_con.reqMarketDataType(data_type)
  return ib_con

# %%
def get_options_data(ib_con, contracts, tick_list="100, 101, 104, 105, 106, 165, 588", signalParameterLive="ask",
                     signalParameterFrozen="last", max_waittime=120):
  contracts = contracts if isinstance(contracts, list) else [contracts]
  ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data
  snapshots = []
  for contract in contracts:
    snapshots.append(ib_con.reqMktData(contract, tick_list, False, False))
  _wait_for_data(ib_con, snapshots, signalParameterFrozen, max_waittime)

  ib_con.reqMarketDataType(1)  # Use free, delayed, frozen data
  _wait_for_data(ib_con, snapshots, signalParameterLive, max_waittime)

  for contract in contracts:
    ib_con.cancelMktData(contract)
  return snapshots


def _wait_for_data(ib_con, objects, indicator_name, max_wait_time):
  wait_time = 0
  while wait_time < max_wait_time and any(
      [ib.util.isNan(getattr(obj, indicator_name)) or getattr(obj, indicator_name) is None for obj in objects]):
    print(f"Waiting {wait_time} / {max_wait_time} for {indicator_name} to be available.")
    ib_con.sleep(1)
    wait_time += 1


def yearly_to_daily_iv(iv):
  if iv is None:
    return None
  return iv / np.sqrt(252)


def third_friday(year, month):
  """Get the third Friday of a specific month in a given year."""
  # Start at the 1st day of the month
  first_day = datetime(year, month, 1)

  # Find the first Friday of the month
  first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)

  # Add 14 days to get the third Friday
  third_friday = first_friday + timedelta(weeks=2)
  return third_friday


# Get the third Fridays of March, June, September, December for a given year
def third_fridays_of_months(year):
  # Relevant months
  months = [3, 6, 9, 12]
  third_fridays = {month: third_friday(year, month) for month in months}
  return third_fridays


def get_and_qualify_contract_details(ib_con, contract):
  details = ib_con.reqContractDetails(contract)
  print(details[0].longName)
  ib_con.qualifyContracts(details[0].contract)
  return details[0]


def get_sigma_move(contract_ticker, sigma, num_days):
  sigma_move = contract_ticker.impliedVolatility * np.sqrt(num_days / 365) * contract_ticker.last
  max_value = np.ceil(contract_ticker.last + sigma * sigma_move)
  min_value = np.floor(contract_ticker.last - sigma * sigma_move)
  return sigma_move, max_value, min_value

def get_last_available(ticker, type: Literal["last", "bid", "ask"] = "last"):
  value = getattr(ticker, type)
  if np.isnan(value) or value is None or value < 0:
    value = getattr(ticker, "prev" + type.capitalize())
  return value

def contract_to_fieldname(contract):
  if contract.secType == 'IND' or contract.secType == 'STK' or contract.secType == 'CMDTY':
    return contract.symbol
  if 'FUT' in contract.secType:
    return f'F{contract.symbol}'
  if contract.secType == 'CASH':
    return contract.symbol + contract.currency

def daily_w_volatility(symbol, type='stk', api='api_paper', offline=False):
  #%%
  contract_filename = f'finance/_data/ibkr/{type}/{symbol}_contract.csv'
  df_existing = None
  if os.path.exists(contract_filename):
    df_existing = pd.read_csv(contract_filename, index_col='date', parse_dates=True)
    df_existing.sort_index(inplace=True)

  if df_existing is not None and not df_existing.empty:
    start_date = df_existing.index[-1] + pd.Timedelta(days=1)
  else:
    start_date = None
#%%
  if offline or (start_date is not None and start_date.date() >= datetime.now().date()):
    return df_existing
#%%
  ib_con = None
  try:
    print(f"{symbol} not found locally, requesting from IBKR...")
    ib_con = utils.ibkr.connect(api, 17, 1)
    contract = ib.Stock(symbol=symbol, exchange='SMART', currency='USD')
    ib_con.qualifyContracts(contract)
    details = ib_con.reqContractDetails(contract)
    print(f"Contract details for {symbol}: {details[0].longName}")

  #%%
    rth = True
    ##%%
    duration = '10 Y'
    barSize = '1 day'
    offset_days = 5
    max_days = 1
    end_date = datetime.now().date() if start_date is None or start_date.date() + timedelta(days=max_days) > datetime.now().date() else start_date + pd.Timedelta(days=max_days)

  # %%
    if start_date is None:
      start_date = ib_con.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=rth, formatDate=True).date()

    def get_request_string(days):
      if days < 365:
        return f'{days+offset_days} D'
      if days < 3650:
        return f'{(days // 365)+1} Y'
      return f'{days // 365} Y'

  #%%
    dfs = [df_existing.copy() if df_existing is not None else pd.DataFrame()]
    while start_date < end_date:
      missing_days = (end_date - start_date).days
      num_days_to_request = 3650 if missing_days > 3650 else missing_days
      start_date = start_date + pd.Timedelta(days=num_days_to_request)
      dfs_types = []
      #%%
      for typ in utils.ibkr.TYPES_OF_DATA[contract.secType]:
        end_date_time_str = start_date.strftime('%Y%m%d %H:%M:%S')
        duration = get_request_string(num_days_to_request)
        data = ib_con.reqHistoricalData(contract, endDateTime=end_date_time_str, durationStr=duration,
                                        barSizeSetting=barSize, whatToShow=typ, useRTH=rth)
        ##%%
        print(f'{datetime.now()} : {utils.ibkr.contract_to_fieldname(contract)} => {typ} {start_date} #{len(data)}')
        if not data:
          print(f'No data for {typ} {contract.symbol}')
          continue
        dfs_type = ib.util.df(data).rename(columns=utils.ibkr.FIELD_NAME_LU[typ])
        dfs_type['date'] = pd.to_datetime(dfs_type['date'])
        dfs_type.set_index('date', inplace=True)
        dfs_types.append(dfs_type)
      dfs_types_merged = pd.concat(dfs_types, axis=1, join='outer')
      for col in ['iv', 'hv']:
        if f'{col}c' in dfs_types_merged.columns:
          dfs_types_merged.drop([f'{col}v', f'{col}bc'], axis=1, inplace=True)
          dfs_types_merged[col] = dfs_types_merged[f'{col}c']
      dfs.append(dfs_types_merged)
    dfs_merged = pd.concat(dfs)
    dfs_merged.sort_index(inplace=True)
    dfs_merged = dfs_merged.loc[~dfs_merged.index.duplicated(keep='last')]
    os.makedirs(os.path.dirname(contract_filename), exist_ok=True)
    dfs_merged.to_csv(contract_filename)
    ib_con.disconnect()
    return dfs_merged
  except Exception as e:
    print(f'Error requesting data for {symbol}: {e}')
    if ib_con is not None and ib_con.isConnected():
      ib_con.disconnect()
    return None
