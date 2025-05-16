# %%
from datetime import datetime
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

from matplotlib.pyplot import tight_layout

import finance.utils as utils

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %%
symbols = ['IBDE40', 'IBES35', 'IBFR40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225']
symbol = symbols[0]
for symbol in symbols:
#%% Create a directory
directory_evals = f'N:/My Drive/Projects/Trading/Research/Strategies/pct_change/{symbol}'
directory_plots = f'N:/My Drive/Projects/Trading/Research/Plots/pct_change/{symbol}'

os.makedirs(directory_evals, exist_ok=True)
os.makedirs(directory_plots, exist_ok=True)

# Create a directory
symbol_def = utils.influx.SYMBOLS[symbol]

exchange = symbol_def['EX']
tz = exchange['TZ']

dfs_ref_range = []
dfs_closing = []
first_day = tz.localize(dateutil.parser.parse('2020-01-02T00:00:00'))
now = datetime.now(tz)
last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

daily_candles = utils.influx.get_candles_range_aggregate(first_day, last_day, symbol, '1d')

daily_candles['pct'] = daily_candles.apply(lambda x: utils.pct.percentage_change(x.o, x.c), axis=1)
#%%
daily_candles.pct.hist(bins=100)
plt.show()

#%%
daily_candles.pct.abs().describe()
