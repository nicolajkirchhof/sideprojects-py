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
contract = Stock('TSLA', 'SMART', 'USD')
available_types_of_data = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY']
types_of_data = ['TRADES', 'BID_ASK', 'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY']
rth=True
data = {}
for typ in types_of_data:
  data[typ] = ib.reqHistoricalData(contract, endDateTime='', durationStr='10 D',
  barSizeSetting='1 day', whatToShow=typ, useRTH=rth)
  print(data[typ])
#%%
dfs = []
for typ in types_of_data:
  df = util.df(data[typ])
  df['name'] = typ
  dfs.append(df)

all_df = pd.concat(dfs)

#%%
# convert to pandas dataframe (pandas needs to be installed):
df = util.df(bars)
df['date'] = pd.to_datetime(df['date'])

#%%
cerebro = bt.Cerebro(preload=True)
data = bt.feeds.PandasData(dataname=df, datetime='date')
cerebro.adddata(data)

# Add the printer as a strategy
cerebro.addstrategy(EmaStrategy)
# cerebro.addstrategy(MyStrategy)
# cerebro.addstrategy(bt.Strategy)

cerebro.run()

cerebro.plot(style='bar', iplot=False)
