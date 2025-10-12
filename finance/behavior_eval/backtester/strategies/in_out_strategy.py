import datetime

import backtrader as bt
from aiohttp.payload import Order

from finance.utils import percentage_change, subtract_percentage

#%%
# Strategy:
# Get in on the first green candle after you have seen at least two red candles with declining VWAP
# Get in after positive gap > 2%
# Stop-Loss is 2%
# Get out if VWAP is not positive anymore
#
OFFSET_PERCENTAGE=2

# Create a Stratey
class InOutStrategy(bt.Strategy):
  def __init__(self):
    print('init')
    bt.ind.MovingAverageSimple(self.data.average, period=1, subplot=False, plotname='VWAP')
    self.sentiment = self.data.close - self.data.open

  def log(self, txt, ts=None):
    ''' Logging function for this strategy'''
    ts = ts or self.datas[0].datetime.datetime(0)
    print(f'{ts}, {txt}')

  def next(self):
    # Current close
    self.log(f'O{self.data.open[0]:.2f} H{self.data.high[0]:.2f} L{self.data.low[0]:.2f} C{self.data.close[0]:.2f}, VWAP{self.data.average[0]:.2f}')

    price = self.data.close[0]
    if self.position:
      # if self.data.average[0] <= self.data.average[-1]:
      #   self.log(f'Close @{self.data.close[0]}')
      #   self.close()
      # elif self.data.average[-1] > self.data.open[0]: # sell immediately
      #   self.close(price=self.data.open[0])
      # else:
        # self.sell(exectype=bt.Order.Stop, price=subtract_percentage(price, 3), valid=self.data.datetime.date()+datetime.timedelta(days=1))
        # stop_price =max(subtract_percentage(price, 2),subtract_percentage(max(self.data.close[0], self.data.open[0], self.data.close[-1]), 0.2))
        stop_price =max(subtract_percentage(price, OFFSET_PERCENTAGE),self.data.low[0])
        self.sell(exectype=bt.Order.Stop, price=stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))
    else:
      # is_gap = percentage_change(self.data.open[0], self.data.close[-1]) > 3
      is_gap = self.data.average[-1] < subtract_percentage(self.data.open[0], 0.25)
      # is_positive_trendchange = self.data.average[0] > self.data.average[-1] #and self.sentiment[0] > 0 > self.sentiment[-1] and self.sentiment[-2] < 0
      is_positive_trendchange = self.data.average[0] > self.data.average[-1] and self.sentiment[0] > 0
      # is_trendchange = False
      if is_gap:
        self.buy(exectype=bt.Order.Market, price=self.data.open[0])
        stop_price =max(subtract_percentage(self.data.open[0], OFFSET_PERCENTAGE),self.data.low[-1])
        self.sell(exectype=bt.Order.Stop, price=stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))
        print(f'\tBUY @{self.data.open[0]} STOP @{stop_price} GAP')
      elif is_positive_trendchange:
        self.buy(exectype=bt.Order.Market, price=self.data.close[0])
        stop_price =max(subtract_percentage(price, OFFSET_PERCENTAGE),self.data.low[0])
        self.sell(exectype=bt.Order.Stop, price=stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))
        print(f'\tBUY @{self.data.close[0]} STOP @{stop_price} TREND')







