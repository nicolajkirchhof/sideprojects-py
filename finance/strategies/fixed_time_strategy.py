import datetime

import backtrader as bt
from aiohttp.payload import Order
from backtrader import Position

from finance.utils import percentage_change, subtract_percentage, add_percentage, profit_loss

#%%
# Strategy:
# LONG
#   Get in on predefined day of week, get out after predefined number of days
#

SIZE=100

# Create a Strategy
class FixedTimeStrategy(bt.Strategy):
  params = (
    ('intervaldays', 25),
    ('buyday', 0), # Monday 0 - Sunday 6
  )

  def __init__(self):
    print('init')
    bt.ind.MovingAverageSimple(self.data.average, period=1, subplot=False, plotname='VWAP')
    self.sentiment = self.data.close - self.data.open
    self.is_long = None
    self.stop_price = None
    self.order_price = None
    self.last_buy_dates = []
    self.last_buy_prices = []
    self.allPL = 0
    self.wins = 0
    self.losses = 0

  def log(self, txt, append=False):
    ''' Logging function for this strategy'''
    ts = self.datas[0].datetime.datetime(0) if not append else ''
    day = f' ({self.datas[0].datetime.datetime(0).weekday()})' if not append else ''
    print(f'{ts}{day}\t {txt}')

  def close_position(self):
    self.log(
      f'Close @{self.data.close[0]:.2f} Size @{self.position.size} P/L {profit_loss(self.position, self.data.close[0]):.2f}',
      True)
    self.close(exectype=bt.Order.Market)

  def next(self):
    # Current close

    # self.log(f'O{self.data.open[0]:.2f} H{self.data.high[0]:.2f} L{self.data.low[0]:.2f} C{self.data.close[0]:.2f}, VWAP{self.data.average[0]:.2f}')

    if self.data.datetime.datetime(0).weekday() == self.params.buyday:
      self.is_long = True
      self.order_price=self.data.open[0]
      self.last_buy_dates.append(self.data.datetime.datetime(0))
      self.last_buy_prices.append(self.order_price)
      self.buy(exectype=bt.Order.Market, price=self.order_price, size=SIZE)

    if len(self.last_buy_dates) > 0 and (self.data.datetime.datetime(0) - self.last_buy_dates[0]).days >= self.params.intervaldays:
      pos = Position(size=SIZE, price=self.last_buy_prices[0])
      pl = profit_loss(pos, self.data.close[0])
      self.allPL += pl
      if pl > 0: self.wins += 1
      else: self.losses += 1

      self.log(
        f'B {self.last_buy_prices[0]:.2f} ({self.last_buy_dates[0].strftime("%Y-%m-%d")}, {self.last_buy_dates[0].weekday()}) S @{self.data.close[0]:.2f} ({self.data.datetime.datetime(0).strftime("%Y-%m-%d")}, {self.data.datetime.datetime(0).weekday()}) P/L {pl:.0f} AP/L {self.allPL:.0f} AW {self.wins} AL {self.losses}',
        True)
      self.last_buy_dates = self.last_buy_dates[1:]
      self.last_buy_prices = self.last_buy_prices[1:]
      self.sell(exectype=bt.Order.Market, size=SIZE)









