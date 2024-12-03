import backtrader as bt

# Create a Stratey
class EmaStrategy(bt.Strategy):
  def __init__(self):
    print('init 5/9 EMA')
    self.sma_5 = bt.ind.EMA(period=5)
    self.sma_9 = bt.ind.EMA(period=9)

  def log(self, txt, ts=None):
    ''' Logging function for this strategy'''
    ts = ts or self.datas[0].datetime.datetime(0)
    print(f'{ts}, {txt}')

  def next(self):
    # Current close
    self.log(f'Open: {self.data.open[-1]:.2f}, {self.data.open[0]:.2f}\
    # Close: {self.data.close[0]:.2f}')

    # if negative candle and open position get out if min is below SMA[-1]
    negative_stop_candle = self.data.close[0] < self.data.open[0] and self.data.close[0] < self.sma_5[-1]
    if self.position and negative_stop_candle:
      self.log(f'SELL @{self.data.close[0]}')
      self.sell()

    positive_entry_candle = self.data.close[0] > self.sma_5[0]
    if not self.position and positive_entry_candle:
      self.log(f'BUY @{self.data.close[0]}')
      self.buy()

    close_below_ema5 = self.data.close[0] < self.sma_5[0]
    # if self
    # if self.close[0] < self.close[-1]:
    #   # current close less than previous close, think about buying
    #   if self.close[-1] < self.close[-2]:
    #     # previous close less than previous close, so buy
    #     self. log('BUY CREATE, .2f' self.close [0])
    #     self.buy()
