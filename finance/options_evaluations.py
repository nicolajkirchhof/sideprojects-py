# %%
import os

import dateutil
import yfinance as yf
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import finplot as fplt
import mplfinance as mpf
import numpy as np

from finance import utils
import blackscholes as bs

mpl.use('TkAgg')
mpl.use('QtAgg')

#%%

stock_names = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'SBUX', 'WDC', 'META', 'NFLX', 'SPY', 'QQQ', 'TQQQ',
               'VTI', 'RSP', 'DOW', 'BA', 'META', 'LUV', 'BKR', 'XOM', 'CVX']
df_contracts = {}
for stock_name in stock_names:
  df_contracts[stock_name] = pd.read_pickle(f'finance/ibkr_finance_data/{stock_name}.pkl')

#%%

df_treasury= pd.read_csv('finance/THREEFY1.csv', parse_dates=['observation_date']).dropna()
#%%
symbol = 'SPY'
date = dateutil.parser.parse('2025-01-24')
# stock_info = df_contracts['TSLA'].tail(1)
stock_info = df_contracts[symbol][df_contracts[symbol].date <= date].tail(1)

# df_treasury[df_treasury['observation_date'] == date]['THREEFY1']

#%%
risk_free_rate_year = df_treasury[df_treasury['observation_date'] <= stock_info.date.iat[0]].tail(1)['THREEFY1'].iat[0]/100

S = stock_info['adj_close'].iat[0]
K = 607
T = 4 / 365
r = risk_free_rate_year
sigma = stock_info['iv_close'].iat[0]*np.sqrt(252)

call = bs.BlackScholesCall(S, K, T, r, sigma)
put = bs.BlackScholesPut(S, K, T, r, sigma)
print(f'call {call.price()} {call.delta()} {call.gamma()} {call.vega()} {call.theta()}')
print(f'put {put.price()} {put.delta()} {put.gamma()} {put.vega()} {put.theta()}')

call = bs.Black76Call(S, K, T, r, sigma)
put = bs.Black76Put(S, K, T, r, sigma)
print(f'call {call.price()} {call.delta()} {call.gamma()} {call.vega()} {call.theta()}')
print(f'put {put.price()} {put.delta()} {put.gamma()} {put.vega()} {put.theta()}')

#%%
import numpy as np
from scipy.stats import norm

N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
  d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
  d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
  d2 = d1 - sigma* np.sqrt(T)
  return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

#%%
print(BS_CALL(S, K, T, r, sigma))
print(BS_PUT(S, K, T, r, sigma))
