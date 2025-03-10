def get_candles_range_aggregate_query(start, end, symbol, group_by_time=None):
  base_query = f'select first(o) as o, last(c) as c, max(h) as h, min(l) as l from {symbol} where time >= \'{start.isoformat()}\' and time < \'{end.isoformat()}\''
  if group_by_time is None:
    return base_query
  return base_query + f' group by time({group_by_time})'

def get_candles_range_raw_query(start, end, symbol):
  return f'select o, c, h, l from {symbol} where time >= \'{start.isoformat()}\' and time < \'{end.isoformat()}\''
