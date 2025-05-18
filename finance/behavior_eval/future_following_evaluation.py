# %%
import datetime
import glob

import dateutil
import numpy as np
import scipy

import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf

import finance.utils as utils
pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
directory = f'N:/My Drive/Trading/Strategies/future_following'
os.makedirs(directory, exist_ok=True)
# symbol = 'IBDE40'
# symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
#            *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]
symbols = ['IBDE40', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100']
symbol = symbols[0]
symbol_directory = f'{directory}/{symbol}'


#%%
# files = glob.glob(f'{symbol_directory}/*2023*.pkl')
# files = glob.glob(f'{symbol_directory}/*2024*.pkl')
files = glob.glob(f'{symbol_directory}/*2025*.pkl')

dfs_follow = []

for file in files:
  dfs_follow.append(pd.read_pickle(file))

##%%
df_follow = pd.concat(dfs_follow)

##%%
df_follow['move'] = df_follow.apply(lambda x: utils.pct.percentage_change(x.low, x.stopout) if x.type == 'long' else utils.pct.percentage_change(x.stopout, x.high), axis=1)
df_follow['loss'] = df_follow.apply(lambda x: x.low > x.stopout if x.type == 'long' else x.stopout > x.high, axis=1)
# %%
# df_follow[df_follow['type'] == 'long'].groupby(['strategy']).agg({'loss':['sum', 'count']})
#%%
# df_follow[(df_follow['type'] == 'long') & (~df_follow['loss'])].groupby(['strategy']).agg({'candles':['max', 'mean', 'std', 'min'], 'move':['max', 'mean', 'std', 'min']})

#%%
# df_follow[df_follow['type'] == 'long'].groupby(['strategy']).agg({'candles':['max', 'mean', 'std', 'min'], 'move':['max', 'mean', 'std', 'min']})
df_follow[df_follow['type'] == 'long'].groupby(['strategy']).agg({'candles':['mean', 'median', 'std'], 'move':['mean', 'median', 'std', 'sum']})


#%%
S_01_pct = '01_pct'
S_02_pct = '02_pct'
S_cbc = 'cbc'
S_cbc_10_pct = 'cbc_10_pct'
S_cbc_20_pct = 'cbc_20_pct'

strategies = [S_01_pct, S_02_pct, S_cbc, S_cbc_10_pct, S_cbc_20_pct]
strategiesToNumber = dict(zip(strategies,  [0, 1, 2, 3, 4]))

df_follow['strategyId'] = df_follow.apply(lambda x: strategiesToNumber[x.strategy], axis=1)


#%%
fig, ax = plt.subplots(2, 3, tight_layout=True, figsize=(24, 13))
axes = ax.flatten()

for i, strategy in enumerate(strategies):
  scatter = df_follow[(df_follow['type'] == 'long') & (df_follow['strategy'] == strategy)].plot.scatter(x='candles', y='move', ax=axes[i])
  axes[i].set_title(f'{symbol} {strategy}')
  plt.show()

#%%
scatter = df_follow[(df_follow['type'] == 'long') & (~df_follow['loss'])].plot.scatter(x='candles', y='move', c='strategyId', colormap='viridis')
# plt.colorbar(scatter.collections[0], label='Z  [S_01_pct, S_02_pct, S_cbc, S_cbc_10_pct, S_cbc_20_pct]')
plt.show()
