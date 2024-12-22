#%%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import scipy
from ib_async import ib, util, IB, Forex, Stock
# util.startLoop()  # uncomment this line when in a notebook
import backtrader as bt

from finance.bull_flag_dip_strategy import BullFlagDipStrategy
from finance.ema_strategy import EmaStrategy
from finance.in_out_strategy import InOutStrategy
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from finance.utils import percentage_change

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%
stock_name = 'TSLA'
df_contract = pd.read_pickle(stock_name + '.pkl')

#%%
ax = df_contract[['average', 'open', 'close']].plot(style='o-')
df_contract[['high', 'low']].plot(ax=ax, style='x')
plt.show()
#%%
plt.figure()
df_pct_vwap = pd.concat([df_contract['average'], df_contract['average'].shift(-1)], axis=1).dropna().apply(lambda x: percentage_change(x.iloc[0], x.iloc[1]), axis=1)
df_pct_vwap.plot()
plt.show()

#%%
def angle_rowwise_v2(A, B):
  p1 = np.einsum('ij,ij->i',A,B)
  p2 = np.einsum('ij,ij->i',A,A)
  p3 = np.einsum('ij,ij->i',B,B)
  p4 = p1 / np.sqrt(p2*p3)
  return np.arccos(np.clip(p4,-1.0,1.0))

#%% Inner angle

df_p2_p3 = pd.concat([df_contract['average'] - df_contract['average'].shift(1), pd.Series(1, df_contract['average'].index, name='X')], axis=1)
df_p2_p1 = df_contract['average'].shift(2) - df_contract['average'].shift(1)
angle_rowwise_v2()
angle.plot()
plt.show()

#%%
df_contract['average'].plot()
X = np.fft.fft(df_contract['average'])
N = len(X)
n = np.arange(N)
T = N/365
freq = n/T
plt.stem(freq, np.abs(X), 'b',
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)
plt.show()
#%%
df_vwap_open_abs = pd.concat([df_contract['average'], df_contract[['open', 'date']].shift(-1)], axis=1).dropna()
df_vwap_open_abs['diff'] = df_vwap_open_abs['average'] - df_vwap_open_abs['open']
df_vwap_open_abs['pct'] = df_vwap_open_abs.apply(lambda x: percentage_change(x.iloc[0], x.iloc[1]), axis=1)
# ax = df_vwap_open_abs.plot(x='date', y=['diff', 'pct'], kind='scatter')
ax = df_vwap_open_abs.plot(x='date', y='pct', kind='scatter')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.show()
#%% match gaussian
# df_vwap_open_abs.hist(column='pct',bins=100)
data = df_vwap_open_abs['pct']
values, bins, bars= plt.hist(data, 100, density=1, alpha=0.5)
mu, sigma = scipy.stats.norm.fit(data)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line)
# plt.bar_label(bars, fontsize=20, color='navy', fmt='{:,.2f}')
loc = mticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
plt.gca().xaxis.set_major_locator(loc)
plt.grid(axis='x', color='0.95')
plt.show()

#get exact values
#%%
x = data.index.values
y = np.array(data)

def lin_interp(x, y, i, half):
  return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
  half = max(y)/2.0
  signs = np.sign(np.add(y, -half))
  zero_crossings = (signs[0:-2] != signs[1:-1])
  zero_crossings_i = np.where(zero_crossings)[0]
  return [lin_interp(x, y, zero_crossings_i[0], half),
          lin_interp(x, y, zero_crossings_i[1], half)]
hmx = half_max_x(x,y)
fwhm = hmx[1] - hmx[0]


#%%
df_vwap_open_pct = df_vwap_open_abs.apply(lambda x: percentage_change(x.iloc[0], x.iloc[1]), axis=1)
df_vwap_open_pct = plot(x='date', y='diff', kind='scatter')
plt.show()
