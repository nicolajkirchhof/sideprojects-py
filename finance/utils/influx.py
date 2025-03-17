import influxdb as idb

from finance.utils.exchanges import DE_EXCHANGE, US_EXCHANGE, GB_EXCHANGE, JP_EXCHANGE, HK_EXCHANGE, AU_EXCHANGE, \
  US_NY_EXCHANGE

DB_INDEX = 'index'
DB_CFD = 'cfd'
DB_FOREX = 'forex'
MPF_COLUMN_MAPPING = ['o', 'h', 'l', 'c', 'v']


SYMBOLS = {'IBDE40': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBNL25': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBCH20': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBES35': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBFR40': {'EX': DE_EXCHANGE, 'DB': DB_CFD},
           'IBGB100': {'EX': GB_EXCHANGE, 'DB': DB_CFD},
           'IBEU50': {'EX': US_EXCHANGE, 'DB': DB_CFD},
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
           'EURUSD': {'EX': US_NY_EXCHANGE, 'DB': DB_FOREX}
           }


def get_influx_clients():
  influx_client_df = idb.DataFrameClient()
  influx_client = idb.InfluxDBClient()

  indices = influx_client.query('show measurements', database=DB_INDEX)
  cfds = influx_client.query('show measurements', database=DB_CFD)
  forex = influx_client.query('show measurements', database=DB_FOREX)

  get_values = lambda x: [y[0] for y in x.raw['series'][0]['values']]
  print('Indices: ', get_values(indices))
  print('Cfds: ', get_values(cfds))
  print('Forex: ', get_values(forex))
  return influx_client_df, influx_client


def get_candles_range_aggregate(start, end, symbol, group_by_time=None):
  symbol_def = SYMBOLS[symbol]
  if symbol_def is None:
    return None
  influx_client_df = idb.DataFrameClient()
  query = get_candles_range_aggregate_query(start, end, symbol, group_by_time)
  influx_data = influx_client_df.query(query, database=symbol_def['DB'])
  if symbol not in influx_data:
    return None
  return influx_data[symbol].tz_convert(symbol_def['EX'].tz)


def get_candles_range_aggregate_query(start, end, symbol, group_by_time=None):
  base_query = f'select first(o) as o, last(c) as c, max(h) as h, min(l) as l from {symbol} where time >= \'{start.isoformat()}\' and time < \'{end.isoformat()}\''
  if group_by_time is None:
    return base_query
  return base_query + f' group by time({group_by_time})'


def get_candles_range_raw_query(start, end, symbol):
  return f'select o, c, h, l from {symbol} where time >= \'{start.isoformat()}\' and time < \'{end.isoformat()}\''
