# %%
from datetime import time
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
import pickle
import numpy as np


import finance.utils as utils

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2
#%%
# TODO:
#


# %%
symbols = ['ESTX50', 'SPX', 'INDU', 'NDX', 'N225']
symbol = symbols[0]
for symbol in symbols:
  #%%
  directory_evals = f'N:/My Drive/Trading/Strategies/close_to_min/{symbol}'
  directory_plots = f'N:/My Drive/Trading/Plots/close_to_min/{symbol}'

  files = glob.glob(f'{directory_evals}/*.pkl')
  #%%
  print(f'Processing {symbol}...')
  results = []
  # Load a pickle file
  for i, file in enumerate(files):
      results.append(pd.read_pickle(file))

  #%%

df_all = pd.concat(results)


#%%

df_all.groupby(['time']).agg({'dist': ['mean', 'median', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25) ]}).plot(kind='bar')

plt.show()

#%%
df_all.groupby(['minT']).agg({'dist': ['mean', 'median', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25) ]}).plot(kind='bar')

plt.show()

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14), tight_layout=True)
df_all[df_all.minT == time(13, 0)].groupby(['time']).agg(
  {'dist': ['mean', 'median', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25) ]}).plot(kind='bar', ax=ax1)

df_all[df_all.minT == time(13, 0)].groupby(['time']).agg({'dist': ['count']}).plot(kind='bar', ax=ax2)
plt.show()

