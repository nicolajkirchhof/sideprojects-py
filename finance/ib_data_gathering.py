#%%
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
%load_ext autoreload
%autoreload 2

#%%
ib.util.startLoop()
ib_con = ib.IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4002
api_paper_port = 4002 
ib_con.connect('127.0.0.1', api_paper_port, clientId=10, readonly=True)
# ib_con.connect('127.0.0.1', tws_paper_port, clientId=10, readonly=True)
ib_con.reqMarketDataType(4)  # Use free, delayed, frozen data

index_client = idb.InfluxDBClient(database='index')
index_client.create_database('index')
index_client_df = idb.DataFrameClient(database='index')
# index_client.create_database('')
# influx schema
# db: TypeOfContract, measure: symbol field: h=high l=low o=open c=close v=volume a=average bc=barCount

#%%
# stock_names = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'SBUX', 'WDC', 'META', 'NFLX', 'SPY', 'QQQ', 'TQQQ', 'VTI', 'RSP', 'DOW']
# stock_names = ['BA', 'META', 'LUV', 'BKR', 'XOM', 'CVX']
eu_indices = [ib.Index(x, 'EUREX', 'EUR') for x in ['DAX', 'ESTX50']]
us_indices = [ib.Index('SPX', 'CBOE', 'USD'), ib.Index('NDX', 'NASDAQ', 'USD'),
              ib.Index('RUT', 'RUSSELL', 'USD'), ib.Index('INDU', 'CME', 'USD')]
fr_index = ib.Index('CAC40', 'MONEP', 'EUR')
es_index = ib.Index('IBEX35', 'MEFFRV', 'EUR')
gb_index = ib.CFD('IBGB100', 'SMART', 'GBP')
jp_index = ib.Index('N225', 'OSE.JPN', 'JPY')
aus_index = ib.CFD('IBAU200', 'SMART', 'AUD')
hk_index = ib.Index('HSI', 'HKFE', 'HKD')

indices = [*eu_indices, *us_indices, gb_index, es_index, jp_index, fr_index, aus_index, hk_index  ]
contracts = indices

for contract in contracts:
  details = ib_con.reqContractDetails(contract)
  print(details[0].longName)
# # # contract_ticker = ib_con.reqMktData(contracts[0], '236, 233, 293, 294, 295, 318, 411, 595', True, False)

#%%
available_types_of_data = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY']
types_of_data = {'IND': ['TRADES'], 'CFD': ['MIDPOINT'] }
rth=False
duration='1 W'
barSize='1 min'
startDateTime=dateutil.parser.parse('2013-06-01')
endDateTime=datetime.datetime.now()

#%%
field_name_lu = {
  'TRADES': {'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v', 'average':'a', 'barCount':'bc' },
  'MIDPOINT': {'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v', 'average':'a', 'barCount':'bc' },
}

# for stock_name in stock_names:
# contract = ib.Index('DAX', 'EUREX', 'EUR')
# contract = contracts[3]
# contract = ib.Contract(symbol='DAX', secType='FUT', exchange='EUREX', currency='EUR', lastTradeDateOrContractMonth='202101')
# details = ib_con.reqContractDetails(contract)
for contract in contracts:
  dfs_contract = {}
  for typ in types_of_data[contract.secType]:
    c_last = index_client_df.query(f'select last("c") from {contract.symbol}')
    if c_last:
      current_date = pd.Timestamp(c_last[contract.symbol].index.values[0]).to_pydatetime()
    else:
      current_date = startDateTime
    while current_date < endDateTime:
      current_date = current_date + pd.Timedelta(weeks=1)
      endDateTimeString = current_date.strftime('%Y%m%d %H:%M:%S')
      data = ib_con.reqHistoricalData(contract, endDateTime=endDateTimeString, durationStr=duration,
      barSizeSetting=barSize, whatToShow=typ, useRTH=rth)
      print(f'{contract.symbol} => {typ} {current_date} #{len(data)}')
      if not data:
        continue
      dfs_type = ib.util.df(data).rename(columns=field_name_lu[typ]).set_index('date')
      index_client_df.write_points(dfs_type, contract.symbol)

#%%
dfs = [dfsc['TRADES'] for dfsc in dfs_contract]
df_contract = pd.concat(dfs)



#%%
for stock_name in stock_names:
  dfs= stock_data[stock_name]
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
