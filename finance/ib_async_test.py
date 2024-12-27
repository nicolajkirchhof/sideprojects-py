#%%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import scipy
from ib_async import ib, util, IB, Forex, Stock
# util.startLoop()  # uncomment this line when in a notebook
import backtrader as bt

from finance.avg_sig_change_strategy import AverageSignificantChangeStrategy
from finance.bull_flag_dip_strategy import BullFlagDipStrategy
from finance.ema_strategy import EmaStrategy
from finance.in_out_strategy import InOutStrategy
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

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

p1_p0 = pd.DataFrame(df_contract['average'] - df_contract['average'].shift(1))
p1_p0['x'] = 1
p1_p0['angle'] = np.arctan2(p1_p0['x'], p1_p0['average'])

p1_p2 = pd.DataFrame(df_contract['average'].shift(2) - df_contract['average'].shift(1))
p1_p2['x'] = -1
p1_p2['angle'] = np.arctan2(p1_p2['x'], p1_p2['average'])

df_contract['deg_change'] = (p1_p0['angle'] - p1_p2['angle'] ) *(180/np.pi)


##%%
class IbkrPandasData(bt.feeds.PandasData):
  lines = ('average','deg_change')
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
    ('deg_change', -1),
  )


#%%
plt.close()
print('=============================== NEW RUN ======================================')
cerebro = bt.Cerebro(preload=True)
data = IbkrPandasData(dataname=df_contract, datetime='date', todate=pd.Timestamp('2024-03-01'))
cerebro.adddata(data)
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=1, name='us', margin=True)
cerebro.broker.set_coc(True)

# Add the printer as a strategy
# cerebro.addstrategy(InOutStrategy)
# cerebro.addstrategy(BullFlagDipStrategy)
cerebro.addstrategy(AverageSignificantChangeStrategy)
# cerebro.addstrategy(EmaStrategy)
# cerebro.addstrategy(MyStrategy)
# cerebro.addstrategy(bt.Strategy)

cerebro.run()
##%%
cerebro.plot(style='bar', iplot=False)
