# %%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
import datetime

import dateutil
import numpy as np
import scipy
# from ib_async import ib
import ib_async as ib
# util.startLoop()  # uncomment this line when in a notebook
import backtrader as bt

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime

import influxdb as idb

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
ib_con.connect('127.0.0.1', api_paper_port, clientId=12, readonly=True)
# ib_con.connect('127.0.0.1', tws_paper_port, clientId=10, readonly=True)
ib_con.reqMarketDataType(2)  # Use free, delayed, frozen data
## %%
influx_client = idb.InfluxDBClient()
influx_client.create_database('index')
influx_client.create_database('future')
influx_client_df = idb.DataFrameClient()
influx_client.create_database('cfd')
influx_client.create_database('forex')
# influx schema

# %%
# stock_names = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'SBUX', 'WDC', 'META', 'NFLX', 'SPY', 'QQQ', 'TQQQ', 'VTI', 'RSP', 'DOW']
# stock_names = ['BA', 'META', 'LUV', 'BKR', 'XOM', 'CVX']
eu_indices = [ib.Index(x, 'EUREX', 'EUR') for x in ['DAX', 'ESTX50']]
us_indices = [ib.Index('SPX', 'CBOE', 'USD'), ib.Index('NDX', 'NASDAQ', 'USD'),
              ib.Index('RUT', 'RUSSELL', 'USD'), ib.Index('INDU', 'CME', 'USD')]
fr_index = ib.Index('CAC40', 'MONEP', 'EUR')
es_index = ib.Index('IBEX35', 'MEFFRV', 'EUR')
jp_index = ib.Index('N225', 'OSE.JPN', 'JPY')
hk_index = ib.Index('HSI', 'HKFE', 'HKD')
indices = [*eu_indices, *us_indices, es_index, jp_index, fr_index, hk_index]
## %%
index_cfd_euro = ['IBGB100', 'IBEU50', 'IBDE40', 'IBFR40', 'IBES35', 'IBNL25', 'IBCH20']
index_cfd_us = ['IBUS500', 'IBUS30', 'IBUST100']
index_cfd_asia = ['IBHK50', 'IBJP225', 'IBAU200']

index_cfds = [ib.CFD(symbol=symbol, exchange='SMART') for symbol in [*index_cfd_euro, *index_cfd_us, *index_cfd_asia]]
# %%

forex = [ib.Forex(symbol=sym, exchange='IDEALPRO', currency=cur) for sym, cur in
         [('EUR', 'USD'), ('EUR', 'GBP'), ('EUR', 'CHF'), ('GBP', 'USD'), ('AUD', 'USD'), ('USD', 'CAD'),
          ('USD', 'JPY'), ('CHF', 'USD')]]

# %%
# eu_futures = [ib.ContFuture(symbol=x, multiplier='1', exchange='EUREX',currency='EUR', includeExpired=True) for x in ['DAX', 'ESTX50']]
# us_futures =[*[ib.ContFuture(symbol=x[0], multiplier=x[1], exchange='CME',currency='USD', includeExpired=True) for x in [('MES', '5'), ('MNQ', '2'), ('RTY', '50')]],
#              ib.ContFuture(symbol='MYM', multiplier='0.5', exchange='CBOT',currency='USD', includeExpired=True),
#             ib.ContFuture(symbol='VXM', multiplier='100', exchange='CFE',currency='USD', includeExpired=True)]
# jp_futures = [ib.ContFuture(symbol='N225M', multiplier='100', exchange='OSE.JPN',currency='JPY', includeExpired=True)]
# swe_futures = [ib.ContFuture(symbol='OMXS30', multiplier='100', exchange='OMS',currency='SEK', includeExpired=True)]
#
# # contracts = indices
# futures = [*eu_futures, *us_futures, *jp_futures, *swe_futures]
# contracts = [*futures, *indices]
contracts = [*forex, *index_cfds, *indices]

for contract in contracts:
  details = ib_con.reqContractDetails(contract)
  print(details[0].longName)
# # # contract_ticker = ib_con.reqMktData(contracts[0], '236, 233, 293, 294, 295, 318, 411, 595', True, False)

# %%
available_types_of_data = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY',
                           'OPTION_IMPLIED_VOLATILITY']
types_of_data = {'IND': ['TRADES'], 'CFD': ['MIDPOINT'], 'FUT': ['TRADES'], 'CONTFUT': ['TRADES'], 'CASH': ['MIDPOINT']}
database_lookup = {'IND': 'index', 'CFD': 'cfd', 'CONTFUT': 'future', 'FUT': 'future', 'CASH': 'forex'}
rth = False
duration = '1 W'
barSize = '1 min'

startDateTime = dateutil.parser.parse('2013-06-01')
# endDateTime=dateutil.parser.parse('2013-06-08')
endDateTime = datetime.datetime.now()


def contract_to_fieldname(contract):
  if contract.secType == 'IND' or contract.secType == 'CFD':
    return contract.symbol
  if 'FUT' in contract.secType:
    return f'F{contract.symbol}'
  if contract.secType == 'CASH':
    return contract.symbol + contract.currency


# %%
field_name_lu = {
  'TRADES': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
  'MIDPOINT': {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v', 'average': 'a', 'barCount': 'bc'},
}

# for stock_name in stock_names:
# details = ib_con.reqContractDetails(contract)
current_date = startDateTime
for contract in contracts:
  dfs_contract = {}
  for typ in types_of_data[contract.secType]:
    c_last = influx_client_df.query(f'select last("c") from {contract_to_fieldname(contract)}',
                                   database=database_lookup[contract.secType])
    if c_last:
      current_date = pd.Timestamp(c_last[contract_to_fieldname(contract)].index.values[0]).to_pydatetime()
    else:
      current_date = startDateTime
    while current_date < endDateTime:
      current_date = current_date + pd.Timedelta(weeks=1)
      endDateTimeString = current_date.strftime('%Y%m%d %H:%M:%S') if not contract.secType == 'CONTFUT' else ""
      data = ib_con.reqHistoricalData(contract, endDateTime=endDateTimeString, durationStr=duration,
                                      barSizeSetting=barSize, whatToShow=typ, useRTH=rth)
      print(f'{contract_to_fieldname(contract)} => {typ} {current_date} #{len(data)}')
      if not data:
        continue
      dfs_type = ib.util.df(data).rename(columns=field_name_lu[typ]).set_index('date')
      influx_client_df.write_points(dfs_type, contract_to_fieldname(contract),
                                   database=database_lookup[contract.secType])

# %%
dfs = [dfsc['TRADES'] for dfsc in dfs_contract]
df_contract = pd.concat(dfs)

# %%
for stock_name in stock_names:
  dfs = stock_data[stock_name]
  df_contract = dfs['TRADES']

  # BID_ASK	Time average bid	Max Ask	Min Bid	Time average ask
  df_contract['ta_bid'] = dfs['BID_ASK'].open
  df_contract['ta_ask'] = dfs['BID_ASK'].close
  df_contract['max_ask'] = dfs['BID_ASK'].high
  df_contract['min_bid'] = dfs['BID_ASK'].low
  df_contract['hvol_avg'] = dfs['HISTORICAL_VOLATILITY'].average
  df_contract['hvol_open'] = dfs['HISTORICAL_VOLATILITY'].open
  df_contract['hvol_close'] = dfs['HISTORICAL_VOLATILITY'].close
  df_contract['iv_avg'] = dfs['OPTION_IMPLIED_VOLATILITY'].average
  df_contract['iv_open'] = dfs['OPTION_IMPLIED_VOLATILITY'].close
  df_contract['iv_close'] = dfs['OPTION_IMPLIED_VOLATILITY'].close
  df_contract['adj_close'] = dfs['ADJUSTED_LAST'].close
  df_contract['adj_open'] = dfs['ADJUSTED_LAST'].open
  df_contract['adj_high'] = dfs['ADJUSTED_LAST'].high
  df_contract['adj_low'] = dfs['ADJUSTED_LAST'].low

  df_contract['date'] = pd.to_datetime(df_contract['date'])

  df_contract.to_pickle(f'finance/ibkr_finance_data/{stock_name}.pkl')
