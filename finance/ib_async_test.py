#%%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
from ib_async import ib, util, IB, Forex, Stock
# util.startLoop()  # uncomment this line when in a notebook
import backtrader as bt


from finance.ema_strategy import EmaStrategy
from finance.in_out_strategy import InOutStrategy
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

from finance.utils import percentage_change

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
stock_name = 'TSLA'
df_contract = pd.read_pickle(stock_name + '.pkl')

#%%
class IbkrPandasData(bt.feeds.PandasData):
  lines = ('average',)
  params = (
    ('datetime', 'timestamp'),
    ('open', -1),
    ('high', -1),
    ('low', -1),
    ('close', -1),
    ('volume', -1),
    ('openinterest',None),
    ('average', -1),
    ('barCount', -1),
    ('ta_bid', -1),
    ('ta_ask', -1),
    ('max_ask', -1),
    ('min_bid', -1),
    ('historical_volatility', -1),
    ('option_implied_volatility', -1),
  )

#%%
df_pct_vwap = pd.concat([df_contract['average'], df_contract['average'].shift(-1)], axis=1).dropna().apply(lambda x: percentage_change(x.iloc[0], x.iloc[1]), axis=1)
df_pct_vwap.plot()
#%%
df_vwap_open_abs = pd.concat([df_contract['average'], df_contract[['open', 'date']].shift(-1)], axis=1).dropna()
df_vwap_open_abs['diff'] = df_vwap_open_abs['average'] - df_vwap_open_abs['open']
df_vwap_open_abs['pct'] = df_vwap_open_abs.apply(lambda x: percentage_change(x.iloc[0], x.iloc[1]), axis=1)
# ax = df_vwap_open_abs.plot(x='date', y=['diff', 'pct'], kind='scatter')
ax = df_vwap_open_abs.plot(x='date', y='pct', kind='scatter')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.show()

#%%
df_vwap_open_pct = df_vwap_open_abs.apply(lambda x: percentage_change(x.iloc[0], x.iloc[1]), axis=1)
df_vwap_open_pct = plot(x='date', y='diff', kind='scatter')
plt.show()
#%%
cerebro = bt.Cerebro(preload=True)
data = IbkrPandasData(dataname=df_contract, datetime='date')
cerebro.adddata(data)

# Add the printer as a strategy
cerebro.addstrategy(InOutStrategy)
# cerebro.addstrategy(EmaStrategy)
# cerebro.addstrategy(MyStrategy)
# cerebro.addstrategy(bt.Strategy)

cerebro.run()

cerebro.plot(style='bar', iplot=False)
