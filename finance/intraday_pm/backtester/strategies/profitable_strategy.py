#%% Cell 1: Import necessary libraries
import backtrader as bt
import yfinance as yf

#%% Cell 2: Define the trading strategy
class SmaRsiStrategy(bt.Strategy):
  def __init__(self):
    print('init 10/30 SMA with RSI')
    self.sma_10 = bt.ind.SMA(period=10)
    self.sma_30 = bt.ind.SMA(period=30)
    self.rsi = bt.ind.RSI(period=14)
  def log(self, txt, ts=None):
    ''' Logging function for this strategy'''
    ts = ts or self.datas[0].datetime.datetime(0)
    print(f'{ts}, {txt}')

  def next(self):
    # Current close
    self.log(f'Open: {self.data.open[-1]:.2f}, {self.data.open[0]:.2f}, Close: {self.data.close[0]:.2f}')

    # Buy condition
    if self.sma_10[0] > self.sma_30[0] and self.sma_10[-1] <= self.sma_30[-1] and self.rsi[0] < 30:
      if not self.position:
        self.log(f'BUY @{self.data.close[0]}')
        self.buy()

    # Sell condition
    if self.sma_10[0] < self.sma_30[0] or self.rsi[0] > 70:
      if self.position:
        self.log(f'SELL @{self.data.close[0]}')
        self.sell()
#%%
if __name__ == '__main__':
#%% Cell 3: Set up the backtrader environment and fetch data
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaRsiStrategy)

    # Fetching data for TSLA
    data = yf.download('TSLA', start='2018-01-01', end='2023-01-01')

    # Cell 4: Add data feed to Cerebro
    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)

    # Cell 5: Set initial cash
    cerebro.broker.setcash(100000.0)

    # Cell 6: Run backtest and print results
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
