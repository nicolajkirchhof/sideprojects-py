import datetime

import backtrader as bt
from aiohttp.payload import Order

from finance.utils import percentage_change, subtract_percentage, add_percentage, profit_loss

#%%
# Strategy:
# LONG
#   Get in if the VWAP increases and the last candle is positive or immediate if the last candle is positive and open > VWAP[-1]
#     Stop Loss is based on the last min - VOLATILITY_MEASURE_STOCK
#   Get out if the last candle is negative and open < VWAP[-1] immediate or on two negative candles in succession
#
# SHORT
#   Reverse of LONG

OFFSET_PERCENTAGE=3

# Create a Strategy
class BullFlagDipStrategy(bt.Strategy):
  def __init__(self):
    print('init')
    bt.ind.MovingAverageSimple(self.data.average, period=1, subplot=False, plotname='VWAP')
    self.sentiment = self.data.close - self.data.open
    self.is_long = None
    self.stop_price = None
    self.order_price = None

  def log(self, txt, append=False):
    ''' Logging function for this strategy'''
    ts = self.datas[0].datetime.datetime(0) if not append else ''
    print(f'{ts}\t {txt}')

  def close_position(self, price=None):
    self.log(f'Close @{price} P/L @{profit_loss(self.order_price, price, self.is_long):.2f}', True)
    self.close(price=price)

  def next(self):
    # Current close
    self.log(f'O{self.data.open[0]:.2f} H{self.data.high[0]:.2f} L{self.data.low[0]:.2f} C{self.data.close[0]:.2f}, VWAP{self.data.average[0]:.2f}')

    if self.position and self.is_long:
      # immediate exit
      if self.stop_price > self.data.low[0]:
        self.log(f'\tStopLossTrigger', True)
        self.close_position(price=self.stop_price)
      elif self.sentiment[-1] < 0 and self.data.open[0] < self.data.average[-1]:
        self.log(f'\tImmediate', True)
        self.close_position(price=self.data.open[0])
      # exit on close
      elif self.sentiment[-1] < 0 and self.sentiment[-2] < 0:
        self.log(f'\tTrend', True)
        self.close_position(price=self.data.close[0])
      else:
        self.stop_price =max(subtract_percentage(self.data.close[0], OFFSET_PERCENTAGE),self.data.low[0])
        self.log(f'SL @{self.stop_price}', True)

    if self.position and not self.is_long:
      # immediate exit
      if self.stop_price < self.data.high[0]:
        self.log(f'\tStopLossTrigger', True)
        self.close_position(price=self.stop_price)
      elif self.sentiment[-1] > 0 and self.data.open[0] > self.data.average[-1]:
        self.log(f'\tImmediate', True)
        self.close_position(price=self.data.open[0])
      # exit on close
      elif self.sentiment[-1] > 0 and self.sentiment[-2] > 0:
        self.log(f'\tTrend', True)
        self.close_position(price=self.data.close[0])
      else:
        self.stop_price =max(add_percentage(self.data.open[0], OFFSET_PERCENTAGE),self.data.high[0])
        self.log(f'SL @{self.stop_price}', True)
      # else:
      #   stop_price =max(subtract_percentage(self.data.close[0], OFFSET_PERCENTAGE),self.data.low[0])
      #   self.log(f'SL @{stop_price}', True)
      #   self.sell(exectype=bt.Order.Stop, price=stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))

    if not self.position:
      # immediate entry
      if self.sentiment[-1] > 0 and self.data.open[0] > self.data.average[-1]:
        self.order_price=self.data.open[0]
        self.buy(exectype=bt.Order.Market, price=self.order_price)
        self.is_long = True
        self.stop_price =max(subtract_percentage(self.data.open[0], OFFSET_PERCENTAGE),self.data.low[-1])
        self.log(f'BUY @{self.data.open[0]} STOP @{self.stop_price} GAP', True)
      elif self.data.average[0] > self.data.average[-1] and self.sentiment[-1] > 0 and self.sentiment[0] > 0:
        self.order_price = self.data.close[0]
        self.buy(exectype=bt.Order.Market, price=self.order_price)
        self.is_long = True
        self.stop_price =max(subtract_percentage(self.data.close[0], OFFSET_PERCENTAGE),self.data.low[0])
        self.log(f'BUY @{self.data.close[0]} STOP @{self.stop_price} TREND', True)
      if self.sentiment[-1] < 0 and self.data.open[0] < self.data.average[-1]:
        self.order_price = self.data.open[0]
        self.sell(exectype=bt.Order.Market, price=self.order_price)
        self.is_long = False
        self.stop_price =min(add_percentage(self.data.open[0], OFFSET_PERCENTAGE),self.data.high[-1])
        self.log(f'SELL @{self.data.open[0]} STOP @{self.stop_price} GAP', True)
      elif self.data.average[0] < self.data.average[-1] and self.sentiment[-1] < 0 and self.sentiment[0] < 0:
        self.order_price = self.data.close[0]
        self.sell(exectype=bt.Order.Market, price=self.order_price)
        self.is_long = False
        self.stop_price =min(add_percentage(self.data.close[0], OFFSET_PERCENTAGE),self.data.high[0])
        self.log(f'SELL @{self.data.close[0]} STOP @{self.stop_price} TREND', True)









