# %%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
from zoneinfo import ZoneInfo

from dateutil import parser
# from ib_async import ib
import ib_async as ib
# util.startLoop()  # uncomment this line when in a notebook

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime
import dateutil

import influxdb as idb
import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
# noinspection PyStatementEffect
%load_ext autoreload
# noinspection PyStatementEffect
%autoreload 2

## %%
ib.util.startLoop()
ib_con = ib.IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4002
api_paper_port = 4002
ib_con.connect('127.0.0.1', api_paper_port, clientId=4, readonly=True)
# ib_con.connect('127.0.0.1', tws_paper_port, clientId=4, readonly=True)
ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data
## %%
influx_client_df, influx_client = utils.influx.get_influx_clients()
influx_client.create_database('future')
# influx schema
##%%
field_name_lu = {
  'TRADES': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'MIDPOINT': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
}
available_types_of_data = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY',
                           'OPTION_IMPLIED_VOLATILITY']
types_of_data = {'IND': ['TRADES'], 'CFD': ['MIDPOINT'], 'FUT': ['TRADES'], 'CONTFUT': ['TRADES'], 'CASH': ['MIDPOINT']}
database_lookup = {'IND': 'index', 'CFD': 'cfd', 'CONTFUT': 'future', 'FUT': 'future', 'CASH': 'forex'}

def contract_to_fieldname(contract):
  if contract.secType == 'IND' or contract.secType == 'CFD':
    return contract.symbol
  if 'FUT' in contract.secType:
    return f'F{contract.symbol}'
  if contract.secType == 'CASH':
    return contract.symbol + contract.currency

## %%
eu_futures = [ib.Future(symbol=x, multiplier='1', exchange='EUREX',currency='EUR', includeExpired=True) for x in ['DAX', 'ESTX50']]
us_futures =[*[ib.Future(symbol=x[0], multiplier=x[1], exchange='CME',currency='USD', includeExpired=True) for x in [('MES', '5'), ('MNQ', '2'), ('RTY', '50')]],
             ib.Future(symbol='MYM', multiplier='0.5', exchange='CBOT',currency='USD', includeExpired=True),
            ib.Future(symbol='VXM', multiplier='100', exchange='CFE',currency='USD', includeExpired=True)]
jp_futures = [ib.Future(symbol='N225M', multiplier='100', exchange='OSE.JPN',currency='JPY', includeExpired=True)]
swe_futures = [ib.Future(symbol='OMXS30', multiplier='100', exchange='OMS',currency='SEK', includeExpired=True)]
#

futures = [*eu_futures, *us_futures, *jp_futures, *swe_futures]

for future in futures:
  details = ib_con.reqContractDetails(future)
  valid_futures = [fut for fut in details if parser.parse(fut.contract.lastTradeDateOrContractMonth) <= datetime.now() + pd.Timedelta(days=90)]
  valid_futures = sorted(valid_futures, key=lambda x: parser.parse(x.contract.lastTradeDateOrContractMonth))

  for valid_future in valid_futures:
    typ = 'TRADES'
    endDateTimeString = parser.parse(valid_future.contract.lastTradeDateOrContractMonth).strftime('%Y%m%d %H:%M:%S')
    future_data = ib_con.reqHistoricalData(valid_future.contract, endDateTime=endDateTimeString, durationStr='365 D',
                                    barSizeSetting='1 day', whatToShow=typ, useRTH=True)
    df_future = ib.util.df(future_data).rename(columns=field_name_lu[typ])
    df_future['date'] = pd.to_datetime(df_future['date'])
    df_future = df_future.set_index('date').tz_localize('UTC')

    ## %%
    rth = False
    ##%%
    duration = '10 D'
    offset_days = 9
    barSize = '1 min'

    startDateTime = df_future.index.min()
    endDateTime = df_future.index.max()
    contract = valid_future.contract

    ## %%
    current_date = startDateTime
    # for stock_name in stock_names:
    for typ in types_of_data[contract.secType]:
      c_last = influx_client_df.query(f'select last("c") from {contract_to_fieldname(contract)}',
                                     database=database_lookup[contract.secType])
      if c_last:
        current_date = pd.Timestamp(c_last[contract_to_fieldname(contract)].index.values[0]).to_pydatetime().replace(tzinfo=ZoneInfo('UTC'))
      else:
        current_date = startDateTime
      while current_date < endDateTime:
        ##%%
        current_date = current_date + pd.Timedelta(days=offset_days)
        endDateTimeString = current_date.strftime('%Y%m%d %H:%M:%S') if not contract.secType == 'CONTFUT' else ""
        data = ib_con.reqHistoricalData(contract, endDateTime=endDateTimeString, durationStr=duration,
                                        barSizeSetting=barSize, whatToShow=typ, useRTH=rth)
        ##%%
        print(f'{datetime.now()} : {contract_to_fieldname(contract)} => {typ} {current_date} #{len(data)}')
        if not data:
          continue
        dfs_type = ib.util.df(data).rename(columns=field_name_lu[typ]).set_index('date').tz_localize('UTC')
        influx_client_df.write_points(dfs_type, contract_to_fieldname(contract),
                                      database=database_lookup[contract.secType])

