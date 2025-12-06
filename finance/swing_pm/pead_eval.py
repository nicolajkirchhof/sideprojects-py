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

import finance.utils as utils

import yfinance as yf
import requests

from finance.swing_pm.earnings_dates import EarningsDates

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%

ed = EarningsDates()
ed.getDateList('MSFT')
# headers = {'User-Agent': 'ResearchApp/1.0 (your_email@example.com)'}
# requests.get('https://www.sec.gov/cgi-bin/browse-edgar?type=10-&dateb=&owner=include&count=100&action=getcompany&CIK=MSFT', headers=headers)


#%%

t = yf.Ticker('MSFT')
t.earnings_history()
