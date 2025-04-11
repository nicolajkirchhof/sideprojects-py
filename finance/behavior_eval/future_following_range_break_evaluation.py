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
from finance.behavior_eval.noon_to_close import pct_change

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
directory = f'N:/My Drive/Projects/Trading/Research/Strategies/future_following_range_break'
os.makedirs(directory, exist_ok=True)
# symbol = 'IBDE40'
# symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
#            *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]
symbols = ['IBDE40', 'IBGB100', 'IBES35', 'IBJP225', 'IBUS30', 'IBUS500', 'IBUST100']
symbol = symbols[0]
symbol_directory = f'{directory}/{symbol}'
S_01_pct = '01_pct'
S_02_pct = '02_pct'
S_cbc = 'cbc'
S_cbc_10_pct = 'cbc_10_pct'
S_cbc_20_pct = 'cbc_20_pct'
S_cbc_10_pct_up = 'cbc_10_pct_up'
S_cbc_20_pct_up = 'cbc_20_pct_up'


#%%

symbol = symbols[1]
timeranges = ['2m', '10m', '5m']
dfs_follow = []
for timerange in timeranges:
  # files = glob.glob(f'{symbol_directory}/*2023*.pkl')
  files = glob.glob(f'{symbol_directory}/*_{timerange}_2024*_follow.pkl')
  # files = glob.glob(f'{symbol_directory}/*2025*.pkl')

  dfs_follow_timerange = []
  for file in files:
    dfs_follow_timerange.append(pd.read_pickle(file))

  ##%%
  dfs_follow_timerange = pd.concat(dfs_follow_timerange)
  dfs_follow_timerange['timerange'] = timerange
  dfs_follow.append(dfs_follow_timerange)
df_follow = pd.concat(dfs_follow)

#%%
def move_max(x):
  if x.loss:
    if x.type == 'long':
      return utils.pct.percentage_change(x.entry, x.stopout)
    else:
      return utils.pct.percentage_change(x.stopout, x.entry)
  else:
    if x.type == 'long':
      return utils.pct.percentage_change(x.entry, x.high)
    else:
      return utils.pct.percentage_change(x.low, x.entry)

df_follow['move'] = df_follow.apply(lambda x: utils.pct.percentage_change(x.entry, x.stopout) if x.type == 'long' else utils.pct.percentage_change(x.stopout, x.entry), axis=1)
df_follow['move_max'] = df_follow.apply(move_max, axis=1)
# df_follow['move_pts'] = df_follow.apply(lambda x:  x.stopout - x.entry - 2 if x.type == 'long' else x.entry - x.stopout -2, axis=1)
#%%
for i in range(1, 5):
  df_follow[f'move_{i}'] = df_follow.apply(lambda x:  utils.pct.percentage_change( x.entry, x[f'high_{i}']) if x.type == 'long' else utils.pct.percentage_change(x[f'low_{i}'], x.entry), axis=1)

# %%
def pct_loss(x):
  return x.sum()/x.count()

# df_follow[df_follow['type'] == 'long'].groupby(['strategy']).agg({'loss':['sum', 'count', pct_loss]})
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'loss':['sum', 'count', pct_loss], }))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'candles':['mean', 'median', 'std']}))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move':['mean', 'median', 'std', 'sum']}))
# print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_max':['mean', 'median', 'std', 'sum']}))
# print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_pts':['mean', 'median', 'std', 'sum']}))

#%%
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move':['mean', 'median', 'std', 'sum']}))
# print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_1':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_2':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_3':['mean', 'median', 'std', 'sum']}))
# print(df_follow.groupby(['timerange', 'strategy', 'type']).agg({'move_4':['mean', 'median', 'std', 'sum']}))
#%%
df_filtred = df_follow[df_follow['timerange'].isin(['2m', '5m', '10m']) & df_follow['strategy'].isin([S_cbc_10_pct, S_cbc, S_cbc_10_pct_up])]
print(df_filtred.groupby(['timerange', 'strategy', 'type']).agg({'move':['sum'], 'move_1':['sum'], 'move_2':['sum'], 'move_3':['sum'], 'move_4':['sum']}))


# df_follow['loss'] = df_follow.apply(lambda x: x.low > x.stopout if x.type == 'long' else x.stopout > x.high, axis=1)
#%%
# print(df_follow.groupby(['timerange', 'strategy', 'type', 'candles']).agg({'loss':['sum', 'count', pct_loss]}))
df_filtred = df_follow[df_follow['timerange'].isin(['2m', '5m']) & df_follow['strategy'].isin([S_cbc_10_pct, S_cbc_10_pct_up])]
print(df_filtred.groupby(['timerange', 'strategy', 'type', 'candles']).agg({'loss':['sum', 'count', pct_loss]}))

#%%
df_filtred = df_follow[df_follow['timerange'].isin(['2m', '5m', '10m']) & df_follow['strategy'].isin([S_cbc_10_pct, S_cbc, S_cbc_10_pct_up])]
df_filtred.groupby(['timerange', 'strategy', 'type', 'candles']).agg({'loss':[pct_loss]}).plot(kind='bar', stacked=True, figsize=(12, 8))
plt.show()

#%%
df_follow[df_follow['loss']].agg({'move':['sum']})
# %%
df_follow.groupby(['strategy', 'type']).agg({'loss':['sum', 'count', pct_loss]})

#%%
# df_follow[(df_follow['type'] == 'long') & (~df_follow['loss'])].groupby(['strategy']).agg({'candles':['max', 'mean', 'std', 'min'], 'move':['max', 'mean', 'std', 'min']})

#%%
# df_follow[df_follow['type'] == 'long'].groupby(['strategy']).agg({'candles':['max', 'mean', 'std', 'min'], 'move':['max', 'mean', 'std', 'min']})
# print(df_follow.groupby(['type','strategy']).agg({'candles':['mean', 'median', 'std'], 'move':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'type','loss', 'strategy']).agg({'candles':['mean', 'median', 'std'], 'move':['mean', 'median', 'std', 'sum']}))
print(df_follow.groupby(['timerange', 'type','loss', 'strategy']).agg({'candles':['mean', 'median', 'std'], 'move_max':['mean', 'median', 'std', 'sum']}))

#%%

print(df_follow.groupby(['timerange', 'type','loss', 'strategy']).agg({'move_max':['mean', 'median', 'std', 'sum']}))

#%%
# strategies = [s_01_pct, s_02_pct, S_cbc, S_cbc_10_pct, S_cbc_20_pct]
strategies = [S_cbc_10_pct, S_cbc, S_cbc_10_pct_up]
strategiesToNumber = dict(zip(strategies,  [0, 1, 2, 3, 4]))
# df_follow['strategyId'] = df_follow.apply(lambda x: strategiesToNumber[x.strategy], axis=1)

for type in ['long', 'short']:
  fig, axs = plt.subplots(len(strategies), len(timeranges), tight_layout=True, figsize=(24, 13))
  fig.suptitle(f'{symbol} {type}')
  for i, timerange in enumerate(timeranges):
    for j, strategy in enumerate(strategies):
      df_follow[(df_follow['type'] == type) & (df_follow['strategy'] == strategy) & (df_follow['timerange'] == timerange)].groupby(['candles']).agg({'loss':pct_change}).plot(kind='bar', x='candles', y='move', ax=axes[i])



#%%
for timerange in timeranges:
  fig, ax = plt.subplots(2, 3, tight_layout=True, figsize=(24, 13))
  fig.suptitle(f'{symbol} {timerange}')
  axes = ax.flatten()
    for i, strategy in enumerate(strategies):
      scatter = df_follow[(df_follow['type'] == 'long') & (df_follow['strategy'] == strategy) & (df_follow['timerange'] == timerange)].plot.scatter(x='candles', y='move', ax=axes[i])
      axes[i].set_title(f'{symbol} {strategy}')
      plt.show()

#%%
scatter = df_follow[(df_follow['type'] == 'long') & (~df_follow['loss'])].plot.scatter(x='candles', y='move', c='strategyId', colormap='viridis')
# plt.colorbar(scatter.collections[0], label='Z  [S_01_pct, S_02_pct, S_cbc, S_cbc_10_pct, S_cbc_20_pct]')
plt.show()
