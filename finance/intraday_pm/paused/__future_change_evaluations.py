# %%
import os

import dateutil
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# import finplot as fplt
import mplfinance as mpf
import numpy as np
from matplotlib.pyplot import tight_layout

from finance import utils, const
import blackscholes as bs

mpl.use('TkAgg')
mpl.use('QtAgg')

%load_ext autoreload
%autoreload 2

#%%

df_contracts = {}
for stock_name in const.stock_names:
  df_contracts[stock_name] = pd.read_pickle(f'finance/ibkr_finance_data/{stock_name}.pkl')
  df_contracts[stock_name].set_index('date', inplace=True)

#%%
start_date = dateutil.parser.parse('2021-01-01')
stock_name = 'T'
df_stk = df_contracts[stock_name]

##%%
df_stk_shift = []
for i in range(0, 46):
  df_stk_shift.append(100*(df_stk[['adj_open', 'adj_close' ]].shift(-i)-df_stk[['adj_open', 'adj_close' ]])/df_stk[['adj_open', 'adj_close' ]])

#%%
fig = plt.figure(tight_layout=True)
days_range = np.arange(0, 46, 2)
for idx, days in enumerate(days_range):
  num_sp = int(np.ceil(np.sqrt(len(days_range))))
  ax = fig.add_subplot(num_sp, num_sp, idx+1)
  df_stk_shift[days+1]['adj_close'].hist(ax = ax, bins=100)
  ax.set_title(f'{days}')

plt.show()

#%%
divider = 2
df_stk_offsets = np.arange(-10, 11, 1)/divider
df_stk_offset_hits_x = []
df_stk_offset_hits_y = []
df_stk_offset_hits_above = []
df_stk_offset_hits_below = []
df_stk_offset_hits_pct = []
for offset in df_stk_offsets:
  print(f'offset {offset}')
  for i in range(0, 46):
     num_above = df_stk_shift[i][(df_stk_shift[i]['adj_close'] >= offset) & (df_stk_shift[i]['adj_close'].index >= start_date )]['adj_close'].count()
     num_below = df_stk_shift[i][(df_stk_shift[i]['adj_close'] < offset) & (df_stk_shift[i]['adj_close'].index >= start_date )]['adj_close'].count()
     df_stk_offset_hits_x.append(i)
     df_stk_offset_hits_y.append(offset)
     df_stk_offset_hits_above.append(num_above)
     df_stk_offset_hits_below.append(num_below)
     df_stk_offset_hits_pct.append(100*num_above/(num_above+num_below))


df_hits = pd.DataFrame({'days': df_stk_offset_hits_x, 'offset': df_stk_offset_hits_y, 'pct': df_stk_offset_hits_pct,'above': df_stk_offset_hits_above, 'below': df_stk_offset_hits_below})
##%%
it_range = np.arange(0, 11,1)/divider
for pn in [-1, 1]:
  fig = plt.figure(tight_layout=True)
  for i, offset in enumerate(it_range):
    ax = fig.add_subplot(len(it_range), 1, i+1)
    df_hits[df_hits['offset'] == pn*offset].plot(ax=ax, kind='bar', x='days', y='pct')
    # ax.set_title(f'{offset}')
    ax.set_ylim(0, 100)
    ax.set_ylabel(f'{pn*offset}')
    for p in ax.patches:
      ax.annotate(f'{p.get_height():.1f}', (p.get_x(), p.get_height()))

plt.show()

