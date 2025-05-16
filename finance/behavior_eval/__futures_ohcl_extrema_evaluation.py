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
import pickle
import numpy as np

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
directory_evals = f'N:/My Drive/Projects/Trading/Research/Strategies/swing_ohcl/{symbol}'
directory_plots = f'N:/My Drive/Projects/Trading/Research/Plots/swing_ohcl/{symbol}'

files = glob.glob(f'{directory_evals}/*.pkl')
#%%
results = []
# Load a pickle file
for i, file in enumerate(files):
  with open(file, 'rb') as f:  # Replace 'file.pkl' with your pickle file path
    data = pickle.load(f)
    data['date'] = data['VWAP'].index[0].date()
    data['VWAP_PCT'] = data['VWAP'].apply(lambda x: utils.pct.percentage_change(x, data['VWAP'].iat[0]))
    data['VWAP_PCT'].index = data['VWAP_PCT'].index.time
    day_change = data['VWAP_PCT'].iat[-1]
    prior_day_change = np.NaN if i < 1 else utils.pct.percentage_change(results[i-1]['VWAP'].iat[-1], data['VWAP'].iat[-1])
    data['day_type'] = 'neutral' if abs(day_change) < 0.5 else 'up' if day_change > 0 else 'down'
    data['prior_day_change'] = prior_day_change
    data['day_change'] = prior_day_change
    results.append(data)

#%%
all_vwap_pct = {}
for day_type in ['up', 'neutral', 'down']:
  all_vwap_pct[day_type] = pd.concat([result['VWAP_PCT'] for result in results if result['day_type'] == day_type])


#%%
# Create the boxplot
fig, axes = plt.subplots(3, 1, figsize=(20, 14), tight_layout=True)  # 3 rows, 1 column

for i, day_type in enumerate(['up', 'neutral', 'down']):
  grouped = all_vwap_pct[day_type].groupby(all_vwap_pct[day_type].index)

  # Prepare the data for the boxplot
  boxplot_data = [group for _, group in grouped]

  axes[i].boxplot(boxplot_data, positions=range(len(grouped)), widths=0.6, patch_artist=True, showfliers=True)
  # Format the x-axis with time labels
  axes[i].set_xticks(range(len(grouped)), [time.strftime('%H:%M') for time in grouped.groups.keys()], rotation=45)
  axes[i].grid(axis='y', linestyle='--', alpha=0.7)
  axes[i].set_ylim(-2, 2)
fig.suptitle("Boxplot of Values Over Time")
# plt.xlabel("Time")
# plt.ylabel("Value")
plt.show()

#%%
# Plot all vwap curves for up, neutral & down days into one chart
year = 2020
fig, axes = plt.subplots(3, 1, figsize=(20, 14), tight_layout=True)  # 3 rows, 1 column
alpha = 0.2
colors = {2020: 'blue', 2021: 'red', 2022: 'green', 2023: 'orange', 2024: 'purple', 2025: 'brown'}

for result in results:
  year = result['date'].year
  day_change_pct = result['VWAP_PCT'].iat[-1]
  if -0.5 < day_change_pct < 0.5:
    result['VWAP_PCT'].plot.line(alpha=alpha, color=colors[year], ax=axes[1])
  elif day_change_pct < -0.5:
    result['VWAP_PCT'].plot.line(alpha=alpha, color=colors[year], ax=axes[0])
  else:
    result['VWAP_PCT'].plot.line(alpha=alpha, color=colors[year], ax=axes[2])

for ax in axes:
  ax.axhline(y=0, color='grey', linestyle='--', linewidth=1, label='Zero Line')
  ax.set_ylim(-2, 2)
plt.show()
