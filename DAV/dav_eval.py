#%%
# from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import scipy

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

df_dav = pd.read_csv('DAV/DAV.csv', parse_dates=['Value date'])
#%%
df_dav_noDiv = df_dav[df_dav['Transaction'] != 'Dividend']
df_dav_noDiv['Year'] = df_dav_noDiv['Value date'].map(lambda x: x.year)
df_dav_noDiv.set_index(['ISIN', 'Value date'], inplace=True)

#%%
df_dav_noDiv_piv = df_dav_noDiv.pivot_table(index=['ISIN', 'Year'], values=['Net amount', 'Quantity'], aggfunc='sum')

#%%
df_dav_noDiv.groupby(df_dav_noDiv['Value date'].map(lambda x: x.year))
