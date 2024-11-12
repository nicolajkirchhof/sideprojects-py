#%%
import yfinance as yf
import pandas as pd
import mplfinance as mpf

import matplotlib.pyplot as plt
%matplotlib qt

#%%
tickers = yf.Tickers('msft')

# access each ticker using (example)
tickers.tickers['MSFT'].info
tickers.tickers['AAPL'].history(period="1mo")
tickers.tickers['GOOG'].actions

tickers.tickers['MSFT'].info
#%%
import finplot as fplt
# import yfinance

# fdf = yfinance.download('AAPL')
fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
fplt.show()
#%%
# data = yf.download("FJTSY", start="2023-11-08", interval="1h")

fjtsy = yf.Ticker('fjtsy')
df = fjtsy.history(start="2024-01-01", interval="1h")
# df = fjtsy.history(start="2024-10-08", interval="1h")

#%%
df['CloseMa7'] = df['Close'].rolling(window=7).mean()
df['HighMa7'] = df['High'].rolling(window=7).mean()
df['LowMa7'] = df['Low'].rolling(window=7).mean()
df['CloseMa21'] = df['Close'].rolling(window=21).mean()
df['HighMa21'] = df['High'].rolling(window=21).mean()
df['LowMa21'] = df['Low'].rolling(window=21).mean()

#%% 
  

df['OrderClosePre']= (df['CloseMa7'].shift(1) > df['CloseMa21'].shift(1)).astype(int)
df['OrderClose'] = (df['CloseMa7'] > df['CloseMa21']).astype(int) - df['OrderClosePre']

buys = df['OrderClose'].where(df['OrderClose'] > 0).multiply(df['Open'])
sels = df['OrderClose'].where(df['OrderClose'] < 0).multiply(df['Open']).multiply(-1)

#%%
# mpf.plot(df, returnfig=True, volume=True, tight_layout=True)
aplt = [mpf.make_addplot(df[['CloseMa7', 'CloseMa21']]), 
        mpf.make_addplot(buys, type='scatter', markersize=100, marker='^'),
        mpf.make_addplot(sels, type='scatter', markersize=100, marker='v')]
#

mpf.plot(df, volume=True, tight_layout=True, addplot=aplt)

#%%
plt.cla()
fig, ax = mpf.plot(df, returnfig=True, volume=True, type='candle', mav=(7,21), tight_layout=True)

df.plot(ax[0], y=['CloseMa', 'HighMa', 'LowMa'])
