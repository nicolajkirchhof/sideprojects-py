import yfinance as yf
import pandas as pd

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
data = yf.download("FJTSY", start="2023-11-08", interval="1h")
#%%
data.plot()

#%%
df = data

df['Close MA'] = df['Close'].rolling(window=7).mean()
