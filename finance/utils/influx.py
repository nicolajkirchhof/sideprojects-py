from datetime import timedelta

import influxdb as idb

from finance.utils.exchanges import DE_EXCHANGE, US_EXCHANGE, GB_EXCHANGE, JP_EXCHANGE, HK_EXCHANGE, AU_EXCHANGE, \
  US_NY_EXCHANGE

DB_INDEX = 'index'
DB_CFD = 'cfd'
DB_FOREX = 'forex'
DB_STK = 'stk'
DB_FUTURE = 'future'
MPF_COLUMN_MAPPING = ['o', 'h', 'l', 'c', 'v']

SEC_TYPE_DB_MAPPING = {
  'IND': DB_INDEX, 'CFD': DB_CFD, 'CONTFUT': DB_FUTURE, 'FUT': DB_FUTURE, 'CASH': DB_FOREX, 'STK': DB_STK, 'CMDTY':DB_CFD
}

SYMBOLS = {'IBDE40': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBNL25': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBCH20': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBES35': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBFR40': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBGB100': {'EX': GB_EXCHANGE, 'DB': DB_CFD},

           'IBEU50': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBUS30': {'EX': US_EXCHANGE, 'DB': DB_CFD},
           'IBUS500': {'EX': US_EXCHANGE, 'DB': DB_CFD},
           'IBUST100': {'EX': US_EXCHANGE, 'DB': DB_CFD},
           'IBJP225': {'EX': JP_EXCHANGE, 'DB': DB_CFD},
           'IBHK50': {'EX': HK_EXCHANGE, 'DB': DB_CFD},
           'IBAU200': {'EX': AU_EXCHANGE, 'DB': DB_CFD},
           'CAC40': {'EX': DE_EXCHANGE, 'DB': DB_INDEX},
           'DAX': {'EX': DE_EXCHANGE, 'DB': DB_INDEX},
           'ESTX50': {'EX': DE_EXCHANGE, 'DB': DB_INDEX},
           'HSI': {'EX': HK_EXCHANGE, 'DB': DB_INDEX},
           'N225': {'EX': JP_EXCHANGE, 'DB': DB_INDEX},
           'NDX': {'EX': US_EXCHANGE, 'DB': DB_INDEX},
           'SPX': {'EX': US_EXCHANGE, 'DB': DB_INDEX},
           'RUT': {'EX': US_EXCHANGE, 'DB': DB_INDEX},
           'INDU': {'EX': US_EXCHANGE, 'DB': DB_INDEX},
           'EURUSD': {'EX': US_NY_EXCHANGE, 'DB': DB_FOREX},
           'USGOLD': {'EX': US_EXCHANGE, 'DB': DB_CFD}
           }


def get_influx_clients():
  influx_client_df = idb.DataFrameClient()
  influx_client = idb.InfluxDBClient()

  indices = influx_client.query('show measurements', database=DB_INDEX)
  cfds = influx_client.query('show measurements', database=DB_CFD)
  forex = influx_client.query('show measurements', database=DB_FOREX)
  stk = influx_client.query('show measurements', database=DB_STK)
  futures = influx_client.query('show measurements', database=DB_FUTURE)

  get_values = lambda x: [y[0] for y in x.raw['series'][0]['values']]
  print('Indices: ', get_values(indices))
  print('Cfds: ', get_values(cfds))
  print('Forex: ', get_values(forex))
  print('Stoks: ', get_values(stk))
  print('Futures: ', get_values(futures))
  return influx_client_df, influx_client


def get_candles_range_aggregate(start, end, symbol, group_by_time=None):
  symbol_def = SYMBOLS[symbol]
  if symbol_def is None:
    return None
  influx_client_df = idb.DataFrameClient()
  query = get_candles_range_aggregate_query(start, end, symbol, group_by_time, with_volatility=symbol_def['DB'] == DB_INDEX)
  influx_data = influx_client_df.query(query, database=symbol_def['DB'])
  if symbol not in influx_data:
    return None
  return influx_data[symbol].tz_convert(symbol_def['EX']['TZ'])

def get_candles_range_raw(start, end, symbol, with_volatility=False):
  symbol_def = SYMBOLS[symbol]
  if symbol_def is None:
    return None
  influx_client_df = idb.DataFrameClient()
  query = get_candles_range_raw_query(start, end, symbol, with_volatility)
  influx_data = influx_client_df.query(query, database=symbol_def['DB'])
  if symbol not in influx_data:
    return None
  return influx_data[symbol].tz_convert(symbol_def['EX']['TZ'])

def get_candles_range_aggregate_query(start, end, symbol, group_by_time=None, with_volatility=False):
  timezone_offset_h = start.utcoffset().total_seconds()/3600
  volatility_query_addon = ', first(hvo) as hvo, last(hvc) as hvc, max(hvh) as hvh, min(hvl) as hvl, first(ivo) as ivo, last(ivc) as ivc, min(ivl) as ivl, max(ivh) as ivh' if with_volatility else ''
  base_query = f'select first(o) as o, last(c) as c, max(h) as h, min(l) as l {volatility_query_addon} from {symbol} where time >= \'{start.isoformat()}\' and time < \'{end.isoformat()}\''
  if group_by_time is None:
    return base_query
  return base_query + f' group by time({group_by_time}, {timezone_offset_h:.0f}h) fill(none)'


def get_candles_range_raw_query(start, end, symbol, with_volatility):
  volatility_query_addon = ', hvo, hvc, hvh, hvl, ivo, ivc, ivl, ivh' if with_volatility else ''
  return f'select o, c, h, l {volatility_query_addon} from {symbol} where time >= \'{start.isoformat()}\' and time < \'{end.isoformat()}\''


def create_databases():
  influx_client = idb.InfluxDBClient()
  influx_client.create_database(DB_INDEX)
  influx_client.create_database(DB_FOREX)
  influx_client.create_database(DB_STK)
  influx_client.create_database(DB_FUTURE)
  influx_client.create_database(DB_CFD)


def sec_type_to_database_name(sec_type):
  return SEC_TYPE_DB_MAPPING[sec_type]

def get_5m_30m_day_date_range_with_indicators(start, end, symbol, cache_offset = timedelta(days=30)):

  return df_5m, df_30m, df_day
