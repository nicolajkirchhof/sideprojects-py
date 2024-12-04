#%%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
from ib_async import ib, util, IB, Forex, Stock
# util.startLoop()  # uncomment this line when in a notebook
import backtrader as bt


from finance.ema_strategy import EmaStrategy
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%
util.startLoop()
ib = IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4002 
api_paper_port = 4002 
ib.connect('127.0.0.1', api_paper_port, clientId=10, readonly=True)
ib.reqMarketDataType(4)  # Use free, delayed, frozen data

#%%
stock_name = 'TSLA'
contract = Stock(stock_name, 'SMART', 'USD')
available_types_of_data = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY']
types_of_data = ['TRADES', 'BID_ASK', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY']
rth=True
data = {}
dfs = {}
for typ in types_of_data:
  data[typ] = ib.reqHistoricalData(contract, endDateTime='', durationStr='1 Y',
  barSizeSetting='1 day', whatToShow=typ, useRTH=rth)
  dfs[typ] = util.df(data[typ])
  print(data[typ])

#%%
df_contract = dfs['TRADES']
# BID_ASK	Time average bid	Max Ask	Min Bid	Time average ask
df_contract['ta_bid'] = dfs['BID_ASK'].open
df_contract['ta_ask'] = dfs['BID_ASK'].close
df_contract['max_ask'] = dfs['BID_ASK'].high
df_contract['min_bid'] = dfs['BID_ASK'].low
df_contract['historical_volatility']= dfs['HISTORICAL_VOLATILITY'].average
df_contract['option_implied_volatility'] =   dfs['OPTION_IMPLIED_VOLATILITY'].average

df_contract['date'] = pd.to_datetime(df_contract['date'])

df_contract.to_pickle(stock_name + '.pkl')
#%%
df_contract = pd.read_pickle(stock_name + '.pkl')
#%%
cerebro = bt.Cerebro(preload=True)
data = bt.feeds.PandasData(dataname=df_contract, datetime='date')
cerebro.adddata(data)

# Add the printer as a strategy
cerebro.addstrategy(EmaStrategy)
# cerebro.addstrategy(MyStrategy)
# cerebro.addstrategy(bt.Strategy)

cerebro.run()

plt.xticks(rotation=45)
cerebro.plot(style='bar', iplot=False)
