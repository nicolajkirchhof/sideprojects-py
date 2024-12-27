import datetime

import backtrader as bt
import numpy as np

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
    self.is_bullish = None
    self.is_bearish = None
    self.stop_price = None
    self.stop_order = None

  def log(self, txt, append=False):
    ''' Logging function for this strategy'''
    ts = self.datas[0].datetime.datetime(0) if not append else ''
    print(f'{ts}\t {txt}')

  def close_position(self):
    self.log(f'Close @{self.data.close[0]:.2f} Size @{self.position.size} P/L {profit_loss(self.position, self.data.close[0]):.2f}', True)
    self.close()

  def next(self):
    if len(self) < 3:
      return

    # Current close
    last_average_change = percentage_change(self.data.average[-2], self.data.average[-1])
    current_average_change = percentage_change(self.data.average[-1], self.data.average[0])
    sign_consistency = np.sign(current_average_change) == np.sign(last_average_change)

    # negative trend sign (short) corresponds with negative position size (short)
    sign_trend_bullish = sign_consistency and np.sign(current_average_change) > 0
    sign_trend_bearish = sign_consistency and np.sign(current_average_change) < 0
    sign_trend_change = sign_trend_bearish and self.position.size > 0 or sign_trend_bullish and self.position.size < 0
    current_sentiment_matches = np.sign(self.sentiment[0]) == np.sign(current_average_change)
    bull_trend_change = self.data.deg_change[0] < 45
    bear_trend_change = self.data.deg_change[0] > (360 - 45)

    self.is_bullish = bull_trend_change or sign_trend_bullish
    self.is_bearish = bear_trend_change or sign_trend_bearish

    position_trend_changed = self.is_bullish and self.position.size < 0 or self.is_bearish and self.position.size > 0

    self.log(f'O{self.data.open[0]:.2f} H{self.data.high[0]:.2f} L{self.data.low[0]:.2f} C{self.data.close[0]:.2f} VWAP{self.data.average[0]:.2f}')
    self.log(f'LAC{last_average_change:.2f} CAC{current_average_change:.2f} DC{self.data.deg_change[0]:.2f}', True)
    self.log(f'SC {sign_consistency} STC {sign_trend_change} CS/NA {current_sentiment_matches} BULL {self.is_bullish} BEAR {self.is_bearish}', True)
    self.log(f'POS {self.position.size} @{self.position.price:.2f} TC {position_trend_changed}', True)
    self.log(f'SO {self.stop_order}', True)

    reverse_position = False
    if self.position and self.position.size > 0:
      # # immediate exit
      if self.stop_price > self.data.low[0]:
        reverse_position = self.is_bearish
        self.log(f'\tSLT {self.stop_price:.2f} REV {reverse_position}', True)
      # if self.sentiment[-1] < 0 and self.data.open[0] < self.data.average[-1]:
      #   self.log(f'\tImmediate', True)
      #   self.close_position(price=self.data.open[0])
      # exit on close
      elif position_trend_changed:
        self.log(f'\tTrend', True)
        self.close_position()
      else:
        tight_stop = max(subtract_percentage(self.data.low[0], OFFSET_PERCENTAGE),self.stop_price)
        self.stop_price = self.stop_price if self.sentiment[0] < 0 else tight_stop
        self.log(f'SL @{self.stop_price:.2f}', True)
        size = self.position.size

        # buy if vwap increases and positive result
        if current_average_change > 0 and self.sentiment[0] > 0:
          self.buy(exectype=bt.Order.Market, size=10,  price=self.data.close[0])
          self.log(f'INC @{self.data.close[0]:.2f}', True)
          size += 10

        self.stop_order = self.sell(exectype=bt.Order.Stop, price=self.stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))

    if self.position and self.position.size < 0:
      if self.stop_price < self.data.high[0]:
        reverse_position = self.is_bullish
        self.log(f'\tSLT {self.stop_price:.2f} REV {reverse_position}', True)
      # elif self.sentiment[-1] > 0 and self.data.open[0] > self.data.average[-1]:
      #   self.log(f'\tImmediate', True)
      #   self.close_position(price=self.data.open[0])
      # exit on close
      elif position_trend_changed:
        self.log(f'\tTrend', True)
        self.close_position()
      else:
        tight_stop =min(add_percentage(self.data.high[0], OFFSET_PERCENTAGE),self.stop_price)
        self.stop_price = self.stop_price if self.sentiment[0] > 0 else tight_stop
        self.log(f'SL @{self.stop_price:.2f}', True)

        size = self.position.size
        if current_average_change < 0 and self.sentiment[0] < 0:
          self.sell(exectype=bt.Order.Market, size=10,  price=self.data.close[0])
          self.log(f'INC @{self.data.close[0]:.2f}', True)
          size += 10

        self.stop_order = self.buy(exectype=bt.Order.Stop, size=size, price=self.stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))

      # else:
      #   stop_price =max(subtract_percentage(self.data.close[0], OFFSET_PERCENTAGE),self.data.low[0])
      #   self.log(f'SL @{stop_price}', True)
      #   self.sell(exectype=bt.Order.Stop, price=stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))

    # in case of reverse position we have not closed with backtrader yet
    if reverse_position or not self.position:
      # immediate entry

      # if self.sentiment[-1] > 0 and self.data.open[0] > self.data.average[-1]:
      #   self.order_price=self.data.open[0]
      #   self.buy(exectype=bt.Order.Market, price=self.order_price)
      #   self.stop_price =max(subtract_percentage(self.data.open[0], OFFSET_PERCENTAGE),self.data.low[-1])
      #   self.log(f'BUY @{self.data.open[0]} STOP @{self.stop_price} GAP', True)
      if self.position.size < 0 or self.is_bullish:
        self.buy(exectype=bt.Order.Market, size=10)
        self.stop_price =subtract_percentage(self.data.low[0], OFFSET_PERCENTAGE)
        self.log(f'BUY @{self.data.close[0]} STOP @{self.stop_price:.2f}', True)
        self.stop_order = self.sell(exectype=bt.Order.Stop, size=10, price=self.stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))
      # if self.sentiment[-1] < 0 and self.data.open[0] < self.data.average[-1]:
      #   self.order_price = self.data.open[0]
      #   self.sell(exectype=bt.Order.Market, price=self.order_price)
      #   self.stop_price =min(add_percentage(self.data.open[0], OFFSET_PERCENTAGE),self.data.high[-1])
      #   self.log(f'SELL @{self.data.open[0]} STOP @{self.stop_price} GAP', True)
      elif self.position.size > 0 or self.is_bearish:
        self.sell(exectype=bt.Order.Market, size=10)
        self.stop_price =add_percentage(self.data.high[0], OFFSET_PERCENTAGE)
        self.log(f'SELL @{self.data.close[0]} STOP @{self.stop_price:.2f}', True)
        self.stop_order = self.buy(exectype=bt.Order.Stop, size=10, price=self.stop_price, valid=self.data.datetime.date()+datetime.timedelta(days=1))









