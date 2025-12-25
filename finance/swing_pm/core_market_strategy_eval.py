#%%
import glob
import pickle
from datetime import datetime, timedelta
from glob import glob
from zoneinfo import ZoneInfo

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
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.mssql.information_schema import columns

import finance.utils as utils

import yfinance as yf
import requests

from finance.swing_pm.earnings_dates import EarningsDates

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%
hist_data_name = '_daily_historical-data'
options_data_name = '_options-overview-history'

#%%
symbol = 'SPY'

files = glob(f'../data/barchart/{hist_data_name}/{symbol}*.pkl')
