import backtrader as bt

# Create a Stratey
class EmaStrategy(bt.Strategy):
  def __init__(self):
    print('init 5/9 EMA')
    self.sma_5 = bt.ind.EMA(period=5)
    self.sma_9 = bt.ind.EMA(period=9)
    self.close = self.datas[0].close

  def log(self, txt, ts=None):
    ''' Logging function for this strategy'''
    ts = ts or self.datas[0].datetime.datetime(0)
    print(f'{ts}, {txt}')

  def next(self):
    # Current close
    self.log(f'Open: {self.datas[0].open[-1]:.2f}, \
    Close: {self.close[0]:.2f}')
    # if self.close[0] < self.close[-1]:
    #   # current close less than previous close, think about buying
    #   if self.close[-1] < self.close[-2]:
    #     # previous close less than previous close, so buy
    #     self. log('BUY CREATE, .2f' self.close [0])
    #     self.buy()
