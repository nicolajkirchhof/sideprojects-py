import datetime

import backtrader as bt
from aiohttp.payload import Order

# Strategy:
# Get in after you have seen at least two red candles with declining VWAP
# Get in after positive gap > 2%
# Stop-Loss is 2%
# Get out if VWAP is not positive anymore
#


# Create a Stratey
class EmaStrategy(bt.Strategy):
  def __init__(self):
    print('init')

  def log(self, txt, ts=None):
    ''' Logging function for this strategy'''
    ts = ts or self.datas[0].datetime.datetime(0)
    print(f'{ts}, {txt}')

  def next(self):
    # Current close
    self.log(f'Open: {self.data.open[0]:.2f},\
    # Close: {self.data.close[0]:.2f}')

    # if negative candle and open position get out if min is below SMA[-1]
    # negative_stop_candle = self.data.close[0] < self.data.open[0] and self.data.close[0] < self.sma_5[-1]
    # if self.position and negative_stop_candle:
    #   self.log(f'Close @{self.data.close[0]}')
    #   self.close()

    positive_entry_candle = self.data.close[0] > self.data.open[0] and self.data.close[0] > self.sma_5[0] > self.sma_20[0]
    new_position = False
    if not self.position and positive_entry_candle:
      self.log(f'BUY @{self.data.close[0]} STOP @{self.sma_5[0]}')
      self.buy(exectype=bt.Order.Close)
      new_position = True


    if self.position or new_position:
      self.sell( exectype=bt.Order.Stop, price=self.sma_5[0], valid=self.data.datetime.date()+datetime.timedelta(days=1))




