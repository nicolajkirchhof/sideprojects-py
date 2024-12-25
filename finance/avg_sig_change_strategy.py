import datetime

import backtrader as bt
import numpy as np
from aiohttp.payload import Order

from finance.utils import percentage_change, subtract_percentage, add_percentage, profit_loss

#%% TODO: Use the inner angle instead of the simple pct addition
#%% Test trailing stop loss based on ATR
#%%
# Strategy:
# LONG
#   Get in if the abs of last two VWAP pct differences is above 4 or if sign of last two pct change is same
#     Stop Loss is based on the last min
#   Get out if the last candle is negative and open < VWAP[-1] immediate or on two negative candles in succession
#
# SHORT
#   Reverse of LONG

OFFSET_PERCENTAGE=0.5
CHANGE_PERCENTAGE=4

# Create a Strategy
class AverageSignificantChangeStrategy(bt.Strategy):
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
    if len(self) < 3:
      return

    # Current close
    last_average_change = percentage_change(self.data.average[-2], self.data.average[-1])
    current_average_change = percentage_change(self.data.average[-1], self.data.average[0])
    sign_consistency = np.sign(current_average_change) == np.sign(last_average_change)

    # Initialize is_long if needed to reverse of current state
    if self.is_long is None and sign_consistency:
      self.is_long = not (np.sign(current_average_change) > 0)


    sign_trend_change = (np.sign(current_average_change) > 0) != self.is_long
    current_sentiment_matches = np.sign(self.sentiment[0]) == np.sign(current_average_change)
    bull_trend_change = self.data.deg_change[0] < 45
    bear_trend_change = self.data.deg_change[0] > (360 - 45)
    is_trend_change = (sign_consistency and sign_trend_change) or (bull_trend_change and not self.is_long) or (bear_trend_change and self.is_long)


    self.log(f'O{self.data.open[0]:.2f} H{self.data.high[0]:.2f} L{self.data.low[0]:.2f} C{self.data.close[0]:.2f} VWAP{self.data.average[0]:.2f}')
    self.log(f'{"LONG" if self.is_long else "SHORT"} LAC{last_average_change:.2f} CAC{current_average_change:.2f} DC{self.data.deg_change[0]:.2f} TC {is_trend_change}', True)
    self.log(f'SC {sign_consistency} ST {sign_trend_change} CS/NA {current_sentiment_matches} BULL {bull_trend_change} BEAR {bear_trend_change}', True)
    self.log(f'POS {self.position.size} @{self.position.price:.2f}', True)

    reverse_position = False
    if self.position and self.is_long:
      # # immediate exit
      if self.stop_price > self.data.low[0]:
        self.close_position(price=self.stop_price)
        reverse_position = self.sentiment[0] < 0
        self.log(f'\tSLT {self.stop_price:.2f} REV {reverse_position}', True)
      # if self.sentiment[-1] < 0 and self.data.open[0] < self.data.average[-1]:
      #   self.log(f'\tImmediate', True)
      #   self.close_position(price=self.data.open[0])
      # exit on close
      elif is_trend_change:
        self.log(f'\tTrend', True)
        self.close_position(price=self.data.close[0])
      else:
        tight_stop = max(subtract_percentage(self.data.low[0], OFFSET_PERCENTAGE),self.stop_price)
        self.stop_price = self.stop_price if self.sentiment[0] < 0 else tight_stop
        self.log(f'SL @{self.stop_price:.2f}', True)

        # buy if vwap increases and positive result
        if current_average_change > 0 and self.sentiment[0] > 0:
          self.buy(exectype=bt.Order.Market, size=10,  price=self.data.close[0])
          self.log(f'INC', True)

    if self.position and not self.is_long:
      if self.stop_price < self.data.high[0]:
        self.close_position(price=self.stop_price)
        reverse_position = self.sentiment[0] > 0
        self.log(f'\tSLT {self.stop_price:.2f} REV {reverse_position}', True)
      # elif self.sentiment[-1] > 0 and self.data.open[0] > self.data.average[-1]:
      #   self.log(f'\tImmediate', True)
      #   self.close_position(price=self.data.open[0])
      # exit on close
      elif is_trend_change:
        self.log(f'\tTrend', True)
        self.close_position(price=self.data.close[0])
      else:
        tight_stop =min(add_percentage(self.data.high[0], OFFSET_PERCENTAGE),self.stop_price)
        self.stop_price = self.stop_price if self.sentiment[0] > 0 else tight_stop
        self.log(f'SL @{self.stop_price:.2f}', True)

        if current_average_change < 0 and self.sentiment[0] < 0:
          self.sell(exectype=bt.Order.Market, size=10,  price=self.data.close[0])
          self.log(f'INC', True)
      # else:
      #   stop_price =max(subtract_percentage(self.data.close[0], OFFSET_PERCENTAGE),self.data.low[0])
      #   self.log(f'SL @{stop_price}', True)
      #   self.sell(exectype=bt.Order.Stop, price=stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))

    # in case of reverse position we have not closed with backtrader yet
    if reverse_position or (not self.position and is_trend_change):
      # immediate entry

      # if self.sentiment[-1] > 0 and self.data.open[0] > self.data.average[-1]:
      #   self.order_price=self.data.open[0]
      #   self.buy(exectype=bt.Order.Market, price=self.order_price)
      #   self.is_long = True
      #   self.stop_price =max(subtract_percentage(self.data.open[0], OFFSET_PERCENTAGE),self.data.low[-1])
      #   self.log(f'BUY @{self.data.open[0]} STOP @{self.stop_price} GAP', True)
      if not self.is_long:
        self.order_price = self.data.close[0]
        self.buy(exectype=bt.Order.Market, size=10,  price=self.order_price)
        self.is_long = True
        self.stop_price =subtract_percentage(self.data.low[0], OFFSET_PERCENTAGE)
        self.log(f'BUY @{self.data.close[0]} STOP @{self.stop_price:.2f} TREND', True)
      # if self.sentiment[-1] < 0 and self.data.open[0] < self.data.average[-1]:
      #   self.order_price = self.data.open[0]
      #   self.sell(exectype=bt.Order.Market, price=self.order_price)
      #   self.is_long = False
      #   self.stop_price =min(add_percentage(self.data.open[0], OFFSET_PERCENTAGE),self.data.high[-1])
      #   self.log(f'SELL @{self.data.open[0]} STOP @{self.stop_price} GAP', True)
      elif self.is_long:
        self.order_price = self.data.close[0]
        self.sell(exectype=bt.Order.Market, size=10, price=self.order_price)
        self.is_long = False
        self.stop_price =add_percentage(self.data.high[0], OFFSET_PERCENTAGE)
        self.log(f'SELL @{self.data.close[0]} STOP @{self.stop_price:.2f} TREND', True)









