#%%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
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
indices = [ib.Index(x) for x in ['DAX','SPX','NDX','RUT', 'RLV', 'RLG','INDU','ESTX50','Z','CAC40','IBEX35']]
contracts = indices

for contract in contracts:
  details = ib_con.reqContractDetails(contract)
  print(details[0].longName)
# # contract_ticker = ib_con.reqMktData(contracts[0], '236, 233, 293, 294, 295, 318, 411, 595', True, False)

#%%
available_types_of_data = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY']
types_of_data = ['TRADES'] #, 'BID','ASK', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY'] #'ADJUSTED_LAST',
rth=False
duration='1 W'
barSize='1 min'
startDateTime=dateutil.parser.parse('2010-01-01')
endDateTime=dateutil.parser.parse('2010-02-01')

#%%
field_name_lu = {
  'TRADES': {'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v', 'average':'a', 'barCount':'bc' },
}


# for stock_name in stock_names:
contract = ib.Index('DAX', 'EUREX', 'EUR')
# contract = ib.Contract(symbol='DAX', secType='FUT', exchange='EUREX', currency='EUR', lastTradeDateOrContractMonth='202101')
# details = ib_con.reqContractDetails(contract)
current_date = startDateTime

dfs_contract = {}
for typ in types_of_data:
  while current_date < endDateTime:
    current_date = current_date + pd.Timedelta(weeks=1)
    endDateTimeString = current_date.strftime('%Y%m%d %H:%M:%S')
    data = ib_con.reqHistoricalData(contract, endDateTime=endDateTimeString, durationStr=duration,
    barSizeSetting=barSize, whatToShow=typ, useRTH=rth)
    dfs_type = ib.util.df(data).rename(columns=field_name_lu[typ]).set_index('date')
    print(f'{contract.symbol} => {typ} {current_date}')

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
