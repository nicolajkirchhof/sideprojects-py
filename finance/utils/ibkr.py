import os

import dateutil
import ib_async as ib
from datetime import datetime, timedelta

import pandas as pd

from finance import utils

AVAILABLE_TYPES_OF_DATA = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY',
                           'OPTION_IMPLIED_VOLATILITY', 'ADJUSTED_LAST']
TYPES_OF_DATA = {'IND': ['TRADES', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY'], 'CFD': ['MIDPOINT'],
                 'FUT': ['TRADES'],
                 'STK': ['TRADES', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY'], 'CONTFUT': ['TRADES'],
                 'CASH': ['MIDPOINT'], 'CMDTY': ['MIDPOINT']}

FIELD_NAME_LU = {
  'TRADES': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'ADJUSTED_LAST': {'open': 'ao', 'high': 'ah', 'low': 'al', 'close': 'ac', 'volume': 'av', 'average': 'aa', 'barCount': 'abc'},
  'MIDPOINT': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'HISTORICAL_VOLATILITY': {'open': 'hvo', 'high': 'hvh', 'low': 'hvl', 'close': 'hvc', 'volume': 'hvv',
                            'average': 'hva', 'barCount': 'hvbc'},
  'OPTION_IMPLIED_VOLATILITY': {'open': 'ivo', 'high': 'ivh', 'low': 'ivl', 'close': 'ivc', 'volume': 'ivv',
                                'average': 'iva', 'barCount': 'ivbc'},
}

NO_OOI_INDICES = ['V2TX', 'V1X', 'VXN', 'RVX', 'VXSLV', 'GVZ', 'OVX']
NO_HV_INDICES = ['VXSLV']
EU_INDICES = [ib.Index(x, 'EUREX', 'EUR') for x in ['DAX', 'ESTX50', 'V2TX', 'V1X']]
US_INDICES = [*[ib.Index(x, 'CBOE', 'USD') for x in ['VIX', 'VXN', 'RVX', 'VXSLV', 'GVZ', 'OVX']],
              ib.Index('SPX', 'CBOE', 'USD'),
              ib.Index('NDX', 'NASDAQ', 'USD'), ib.Index('RUT', 'RUSSELL', 'USD'),
              ib.Index('INDU', 'CME', 'USD')]
FR_INDEX = ib.Index('CAC40', 'MONEP', 'EUR')
# es_index = ib.Index('IBEX35', 'MEFFRV', 'EUR')
JP_INDEX = ib.Index('N225', 'OSE.JPN', 'JPY')
HK_INDEX = ib.Index('HSI', 'HKFE', 'HKD')
INDICES = [*EU_INDICES, *US_INDICES, JP_INDEX, FR_INDEX, HK_INDEX]

def cache_path(symbol: str) -> str:
  """Return the on-disk pickle path used by `daily_w_volatility` for this symbol."""
  if '.' in symbol:
    symbol = symbol.replace('.', ' ')
  return f'finance/_data/ibkr/{symbol.replace(" ", "_")}_contract.pkl'


def get_cached_last_bar_date(symbol: str):
  """Return the last cached bar date for `symbol`, or None if no cache exists."""
  path = cache_path(symbol)
  if not os.path.exists(path):
    return None
  try:
    df = pd.read_pickle(path)
    if df is None or df.empty:
      return None
    return df.index.max().date()
  except Exception:
    return None


def is_cache_fresh(symbol: str, max_age_days: int = 1) -> bool:
  """True if cache exists and its last bar is within `max_age_days` of today."""
  last = get_cached_last_bar_date(symbol)
  if last is None:
    return False
  return (datetime.now().date() - last).days <= max_age_days


def has_cache(symbol: str) -> bool:
  """True if a cache pickle exists for `symbol` (may be stale)."""
  return os.path.exists(cache_path(symbol))


def connect(instance, id, data_type):
  ib.util.startLoop()
  ib_con = ib.IB()
  tws_ports = {'real': 8497, 'paper': 7498, 'api': 4001, 'api_paper': 4002}
  ib_con.connect('127.0.0.1', tws_ports[instance], clientId=id, readonly=True)
  ib_con.reqMarketDataType(data_type)
  return ib_con

def contract_to_fieldname(contract):
  if contract.secType == 'IND' or contract.secType == 'STK' or contract.secType == 'CMDTY':
    return contract.symbol
  if 'FUT' in contract.secType:
    return f'F{contract.symbol}'
  if contract.secType == 'CASH':
    return contract.symbol + contract.currency

def daily_w_volatility(symbol, api='api_paper', offline=False, ib_con=None, refresh_offset_days=5):
  #%%
  if '.' in symbol:
    symbol = symbol.replace('.', ' ')
  contract_filename = f'finance/_data/ibkr/{symbol.replace(" ", "_")}_contract.pkl'
  df_existing = None
  if os.path.exists(contract_filename):
    df_existing = pd.read_pickle(contract_filename)
    df_existing.sort_index(inplace=True)

  if df_existing is not None and not df_existing.empty:
    # start from the day after the last cached candle
    cursor = df_existing.index[-1].date() + timedelta(days=1)
  else:
    cursor = None

  if offline or (cursor is not None and cursor > datetime.now().date() - timedelta(days=refresh_offset_days)):
    return df_existing

  disconnect = False
  try:
    if ib_con is None:
      print(f"{symbol} not found locally, requesting from IBKR...")
      ib_con = utils.ibkr.connect(api, 17, 1)
      disconnect = True

    if symbol.startswith('^'):
      contract = ib.Forex(symbol=symbol[1:4], exchange='IDEALPRO', currency=symbol[4:])
    if symbol.startswith('$$'):
      contract = ib.CFD(symbol=symbol.replace('$$', ''), exchange='SMART')
    if symbol.startswith('$'):
      contract = [x for x in INDICES if x.symbol == symbol[1:]][0]
    else:
      contract = ib.Stock(symbol=symbol, exchange='SMART', currency='USD')
    ib_con.qualifyContracts(contract  )
    details = ib_con.reqContractDetails(contract)
    print(f"Contract details for {symbol}: {details[0].longName}")

  #%%
    rth = True
    ##%%
    duration = '10 Y'
    barSize = '1 day'
    offset_days = 5

    # Always try to catch up to "today"
    target_end = datetime.now().date()

    if cursor is None:
     cursor = ib_con.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=rth, formatDate=True)
     if cursor is None or cursor is not datetime:
       cursor = dateutil.parser.parse("2000-01-01").date()

    # If already up-to-date (cache past today), return cache
    if cursor > target_end:
      return df_existing

    def get_request_string(days: int) -> str:
      # IBKR durationStr is inclusive-ish; add a small cushion
      days = max(int(days), 1)
      if days < 365:
        return f'{days + offset_days} D'
      if days < 3650:
        return f'{(days // 365) + 1} Y'
      return f'{days // 365} Y'

    # Request in chunks to avoid giant single calls
    max_chunk_days = 365  # tune if you want (e.g. 180 or 730)

    dfs = [df_existing.copy() if df_existing is not None else pd.DataFrame()]

    while cursor <= target_end:  # inclusive — always try to include `today`

      request_end = min(cursor + timedelta(days=max_chunk_days), target_end)
      days_to_request = (request_end - cursor).days + 1  # include end day

      dfs_types = []
      for typ in utils.ibkr.TYPES_OF_DATA[contract.secType]:
        if typ == 'OPTION_IMPLIED_VOLATILITY' and contract.symbol in NO_OOI_INDICES \
            or typ == 'HISTORICAL_VOLATILITY' and contract.symbol in NO_HV_INDICES:
          continue

        end_date_time_str = request_end.strftime('%Y%m%d %H:%M:%S')
        duration = get_request_string(days_to_request)

        data = ib_con.reqHistoricalData(
          contract,
          endDateTime=end_date_time_str,
          durationStr=duration,
          barSizeSetting=barSize,
          whatToShow=typ,
          useRTH=rth
        )

        print(f'{datetime.now()} : {utils.ibkr.contract_to_fieldname(contract)} => {typ} {cursor}..{request_end} #{len(data)}')
        if not data:
          print(f'No data for {typ} {contract.symbol}')
          continue

        dfs_type = ib.util.df(data).rename(columns=utils.ibkr.FIELD_NAME_LU[typ])
        dfs_type['date'] = pd.to_datetime(dfs_type['date'])
        dfs_type.set_index('date', inplace=True)
        dfs_types.append(dfs_type)

      if dfs_types:
        dfs_types_merged = pd.concat(dfs_types, axis=1, join='outer')
        for col in ['iv', 'hv']:
          if f'{col}c' in dfs_types_merged.columns:
            dfs_types_merged.drop([f'{col}v', f'{col}bc'], axis=1, inplace=True)
            dfs_types_merged[col] = dfs_types_merged[f'{col}c']
        dfs.append(dfs_types_merged)

      # advance cursor to the day after request_end
      cursor = request_end + timedelta(days=1)

    dfs_merged = pd.concat(dfs)
    dfs_merged.sort_index(inplace=True)
    dfs_merged = dfs_merged.loc[~dfs_merged.index.duplicated(keep='last')]

    os.makedirs(os.path.dirname(contract_filename), exist_ok=True)
    dfs_merged.to_pickle(contract_filename)

    if disconnect:
      ib_con.disconnect()
    return dfs_merged

  except Exception as e:
    print(f'IBKR: Error requesting data for {symbol}: {e}')
    if ib_con is not None and disconnect and ib_con.isConnected():
      ib_con.disconnect()
    return None
