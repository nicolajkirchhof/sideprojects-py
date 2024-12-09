import datetime

import backtrader as bt
from aiohttp.payload import Order

from finance.utils import percentage_change, subtract_percentage


# Strategy:
# Get in on the first green candle after you have seen at least two red candles with declining VWAP
# Get in after positive gap > 2%
# Stop-Loss is 2%
# Get out if VWAP is not positive anymore
#


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
    self.log(f'Open: {self.data.open[0]:.2f}, Close: {self.data.close[0]:.2f}, VWAP: {self.data.average[0]:.2f}')

    price = self.data.close[0]
    if self.position:
      # if self.data.average[0] <= self.data.average[-1]:
      #   self.log(f'Close @{self.data.close[0]}')
      #   self.close()
      # elif self.data.average[-1] > self.data.open[0]: # sell immediately
      #   self.close(price=self.data.open[0])
      # else:
        # self.sell(exectype=bt.Order.Stop, price=subtract_percentage(price, 3), valid=self.data.datetime.date()+datetime.timedelta(days=1))
        stop_price =max(subtract_percentage(price, 3),subtract_percentage(max(self.data.close[0], self.data.open[0], self.data.close[-1]), 0.5))
        self.sell(exectype=bt.Order.Stop, price=stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))
    else:
      # is_gap = percentage_change(self.data.open[0], self.data.close[-1]) > 3
      is_gap = self.data.average[-1] < subtract_percentage(self.data.open[0], 0.25)
      is_trendchange = self.data.average[0] > self.data.average[-1] #and self.sentiment[0] > 0 > self.sentiment[-1] and self.sentiment[-2] < 0
      # is_trendchange = False
      if is_trendchange or is_gap:
        price = self.data.close[0] if is_trendchange else self.data.open[0]
        self.buy(exectype=bt.Order.Market, price=price)
        stop_price =max(subtract_percentage(price, 3),subtract_percentage(max(self.data.close[0], self.data.open[0]),0.5))
        self.sell(exectype=bt.Order.Stop, price=stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))
        self.log(f'BUY @{price} STOP @{stop_price} GAP {is_gap} TREND {is_trendchange}')

    # if negative candle and open position get out if min is below SMA[-1]
    # negative_stop_candle = self.data.close[0] < self.data.open[0] and self.data.close[0] < self.sma_5[-1]
    # if self.position and negative_stop_candle
    #   self.log(f'Close @{self.data.close[0]}')
    #   self.close()

    # positive_entry_candle = self.data.close[0] > self.data.open[0] and self.data.close[0] > self.sma_5[0] > self.sma_20[0]
    # new_position = False
    # if not self.position and positive_entry_candle:
    #   self.log(f'BUY @{self.data.close[0]} STOP @{self.sma_5[0]}')
    #   self.buy(exectype=bt.Order.Close)
    #   new_position = True
    #
    #
    # if self.position or new_position:
    #   self.sell( exectype=bt.Order.Stop, price=self.sma_5[0], valid=self.data.datetime.date()+datetime.timedelta(days=1))






