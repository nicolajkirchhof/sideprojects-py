#%%
from datetime import datetime, timedelta


import dateutil
import numpy as np
import scipy

import pandas as pd
import os

import influxdb as idb

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
from matplotlib import gridspec

import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
symbols = ['IBDE40', 'IBES35', 'IBFR40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225']
#symbol = symbols[0]
for symbol in symbols[-1:]:
  #%% Create a directory
  directory = f'N:/My Drive/Trading/Plots/5m_30m_d_w/{symbol}'
  os.makedirs(directory, exist_ok=True)

  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]

  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = dateutil.parser.parse('2020-01-02T00:00:00').replace(tzinfo=tz)
  # first_day = dateutil.parser.parse('2025-03-06T00:00:00').replace(tzinfo=tz)
  now = datetime.now(tz)
  last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  #%%
  while day_start < last_day:
    #%%
    day_data = utils.trading_day_data.TradingDayData(day_start, symbol)

    date_str = day_data.day_start.strftime('%Y-%m-%d')
    utils.plots.daily_change_plot(day_data)

    # plt.show()
    ##%%
    plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
    plt.close()
    print(f'{symbol} finished {date_str}')
    ##%%
    prior_day = day_start
    day_start = day_start + timedelta(days=1)

