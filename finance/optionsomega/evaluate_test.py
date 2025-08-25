#%%
import pickle
from datetime import datetime, timedelta, time
from time import sleep

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
from scipy.spatial import ConvexHull

import finance.utils as utils
import yfinance as yf

pd.options.plotting.backend = "matplotlib"
pd.set_option("display.max_columns", None)  # None means "no limit"
pd.set_option("display.max_rows", None)  # None means "no limit"
pd.set_option("display.width", 140)  # Set the width to 80 characters


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%
# Download VIX data
df_vix = pd.read_csv('finance/_data/VIX_History.csv', parse_dates=True,index_col=['DATE'])
df_vix = df_vix.rename(columns={'OPEN': 'vo', 'HIGH': 'vh', 'LOW': 'vl', 'CLOSE': 'vc'})
df_vix['vhcl'] = df_vix[['vh', 'vc', 'vl']].sum(axis=1)/3

#%%
directory = f'N:/My Drive/Trading/Strategies/OptionOmega'

# file = f'{directory}/60-DTE-111-Early-Exit-14-DTE-10D-30D-5D.csv'
# file = f'{directory}/SPY-60DTE-111-10D-30D-5D-5.csv'
# file = f'{directory}/07-DTE-Naket-put-07D-AM-SL200.csv'
# file = f'{directory}/07-DTE-Naket-put-05D-AM-SL200.csv'
file = f'{directory}/56-DTE-Naket-put-10D-AM-SL200.csv'

df = pd.read_csv(file)
df['DateTimeOpened'] = pd.to_datetime(df['Date Opened']+ 'T' +df['Time Opened']).dt.tz_localize('America/New_York')
df['DateTimeClosed'] = pd.to_datetime(df['Date Closed']+ 'T' +df['Time Closed']).dt.tz_localize('America/New_York')
df['Date Opened'] = pd.to_datetime(df['Date Opened'])
df['Date Closed'] = pd.to_datetime(df['Date Closed'])
df['Weekday'] = df['Date Opened'].dt.strftime('%a')
df['Month'] = df['Date Opened'].dt.strftime('%b')
df['Year'] = df['Date Opened'].dt.strftime('%Y')
df['W/L'] = df['P/L'] > 0

#%%
df_v = pd.merge(df, df_vix,
                     left_on='Date Opened',
                     right_on='DATE',
                     how='inner')
df_v.set_index('DateTimeOpened', inplace=True)
df_v.sort_index(inplace=True)
#%%
df_v[df_v['P/L'] < 0][['vh', 'vl']].plot()
plt.show()
df_v[df_v['P/L'] > 0][['vh', 'vl']].plot()
plt.show()

#%%
df_v.plot.scatter(x='vhcl', y='P/L', style='o')
plt.show()
df_v[['vhcl', 'P/L']].corr()
#%%
df_v['VixRegime'] = pd.cut(df_v['vhcl'], bins=[-np.inf, 13, 25, np.inf], labels=['v<13', '13<v<25', 'v>25' ])
df_v.groupby(['VixRegime']).agg({'P/L':['sum'], 'W/L':['sum' ,'count', lambda x: x.sum()/x.count()]})
#%%
print(df_v.groupby(['Weekday']).agg({'P/L':['sum']}))
df_v.plot.scatter(x='Weekday', y='P/L', style='o')
plt.show()


df_v.groupby(['Year', 'Weekday']).agg({'P/L':['sum']})
