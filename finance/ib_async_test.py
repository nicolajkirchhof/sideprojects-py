#%%
from __future__ import (absolute_import, division, print_function, unicode_literals)
from ib_async import *
# util.startLoop()  # uncomment this line when in a notebook
import backtrader as bt
import matplotlib.pyplot as plt

from ema_strategy import EmaStrategy
%matplotlib qt

#%%
util.startLoop()
ib = IB()
tws_real_port = 7497
tws_paper_port = 7497
api_real_port = 4002 
api_paper_port = 4002 
ib.connect('127.0.0.1', api_paper_port, clientId=1, readonly=True)

ib.reqMarketDataType(4)  # Use free, delayed, frozen data
contract = Forex('EURUSD')
bars = ib.reqHistoricalData(
  contract, endDateTime='', durationStr='30 D',
  barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

# convert to pandas dataframe (pandas needs to be installed):
df = util.df(bars)
print(df)


#%%
cerebro = bt.Cerebro()
data = bt.feeds.PandasData(dataname=df, datetime='date')
cerebro.adddata(data)

# Add the printer as a strategy
cerebro.addstrategy(EmaStrategy)
# cerebro.addstrategy(bt.Strategy)

cerebro.run()

cerebro.plot(style='bar', iplot=False)
