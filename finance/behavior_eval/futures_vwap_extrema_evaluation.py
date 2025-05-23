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
#%%
# TODO:
# - Determine time between Significant points
# - How does it change in comparison to the last hour before the opening. Is that a good setup?
# - Analyze the changes thoughout the day, how many percentages are made where
#   - What might be triggers for a positive/negative rally into the close of the US indices?
# -
# %%
symbols = ['IBDE40', 'IBES35', 'IBFR40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225']
symbol = symbols[0]
for symbol in symbols:
  #%% Create a directory
  directory_evals = f'N:/My Drive/Trading/Strategies/swing_vwap/{symbol}'
  directory_plots = f'N:/My Drive/Trading/Plots/swing_vwap/{symbol}_eval'
  os.makedirs(directory_plots, exist_ok=True)

  files = glob.glob(f'{directory_evals}/*.pkl')
  #%%
  print(f'Processing {symbol}...')
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
      data['day_type_prior'] = 'neutral' if abs(prior_day_change) < 0.5 else 'up' if prior_day_change > 0 else 'down'
      data['day_change'] = prior_day_change
      results.append(data)

  #%%
  # Evaluate the time of the extrema
  extrema = [data['extrema'] for data in results]
  df_extrema = pd.concat(extrema)
  df_extrema.set_index('ts', inplace=True)

  df_extrema['time'] = df_extrema.index.time
  df_extrema['date'] = df_extrema.index.date

  df_extrema_lhd = df_extrema[~df_extrema.type.isin(['o', 'c'])].copy()

  df_extrema_lh = df_extrema[df_extrema.type.isin(['l', 'h'])].copy()
  df_extrema_lh['is_first_of_day'] = df_extrema_lh.groupby('date')['time'].transform('first') == df_extrema_lh['time']

  df_extrema_d = df_extrema[df_extrema.type.isin(['dev'])].copy()

  #%%
  df = df_extrema_lhd
  def plot_type_percentages(df, event = ''):
    # Calculate percentages for each type independently
    type_counts = df.groupby(['time', 'type']).size().unstack(fill_value=0)
    type_percentages = type_counts.div(type_counts.sum(), axis=1) * 100

    # Calculate percentages for each type independently
    all_counts = df.groupby(['time']).size()
    all_percentages = all_counts.div(all_counts.sum()) * 100
    type_percentages['all'] = all_percentages

    # Calculate cumulative percentages for each type independently
    cumulative_counts = type_counts.cumsum()
    cumulative_percentages = cumulative_counts.div(type_counts.sum(), axis=1) * 100

    all_cumulative_counts = all_counts.cumsum()
    all_cumulative_percentages = all_cumulative_counts.div(all_counts.sum()) * 100
    cumulative_percentages['all'] = all_cumulative_percentages

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14))

    # Plot 1: Individual type percentages
    type_percentages.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title(f'{symbol} Distribution of {event} Events by Type\n(Each type adds to 100%)')
    ax1.set_xlabel('')
    ax1.set_ylabel('Percentage of Type Events')
    ax1.legend(title='Type')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add percentage labels on the bars
    # for container in ax1.containers:
    #   ax1.bar_label(container, fmt='%.1f%%', padding=3)

    # Plot 2: Cumulative percentages
    cumulative_percentages.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title(f'{symbol} Cumulative Distribution of {event} Events by Type\n(Each type reaches 100%)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Percentage of Type Events')
    ax2.legend(title='Type')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add percentage labels on the bars
    # for container in ax2.containers:
    #   ax2.bar_label(container, fmt='%.1f%%', padding=3)

    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout to prevent overlap
    plt.tight_layout()

  #%%
  print('Plotting...')
  plot_type_percentages(df_extrema_lh[df_extrema_lh.is_first_of_day], 'First of Day')
  plt.savefig(f'{directory_plots}/{symbol}_FirstOfDay.png', bbox_inches='tight')  # High-quality save
  plt.close()
  plot_type_percentages(df_extrema_lh[~df_extrema_lh.is_first_of_day], 'Second of Day')
  plt.savefig(f'{directory_plots}/{symbol}_SecondOfDay.png', bbox_inches='tight')  # High-quality save
  plt.close()
  plot_type_percentages(df_extrema_d, 'Deviation')
  plt.savefig(f'{directory_plots}/{symbol}_Deviations.png', bbox_inches='tight')  # High-quality save
  plt.close()


  #%%
  all_vwap_pct = {}
  for day_type in ['up', 'neutral', 'down']:
    # all_vwap_pct[day_type] = pd.concat([result['VWAP_PCT'] for result in results if result['day_type'] == day_type])
    all_vwap_pct[day_type] = pd.concat([result['VWAP_PCT'] for result in results if result['day_type_prior'] == day_type])


  #%%
  # Create the boxplot
  fig, axes = plt.subplots(3, 1, figsize=(20, 14), tight_layout=True)  # 3 rows, 1 column

  for i, day_type in enumerate(['up', 'neutral', 'down']):
    grouped = all_vwap_pct[day_type].groupby(all_vwap_pct[day_type].index)

    # Prepare the data for the boxplot
    boxplot_data = [group for _, group in grouped]

    axes[i].boxplot(boxplot_data, positions=range(len(grouped)), widths=0.6, patch_artist=True, showfliers=False, notch=True)
    # axes[i].violinplot(boxplot_data, positions=range(len(grouped)), widths=0.6, showmeans=True, showmedians=True, showextrema=False)
    # Format the x-axis with time labels
    axes[i].set_xticks(range(len(grouped)), [time.strftime('%H:%M') for time in grouped.groups.keys()], rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    # axes[i].set_ylim(-2, 2)
  fig.suptitle("Boxplot of VWAP percentage changes")
  # plt.show()
  plt.savefig(f'{directory_plots}/{symbol}_VWAP_PCT.png', bbox_inches='tight')  # High-quality save
  plt.close()

  print('Done.')




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



### Plot of counts over time, not so useful
#%%
def plot_type_counts(df, suptitle = ''):
  count_df = df.groupby(['time', 'type']).size().unstack(fill_value=0)

  # Create the bar chart
  ax = count_df.plot(kind='bar', figsize=(24, 14))

  # Customize the chart
  plt.title('Frequency of Types Over Time')
  plt.suptitle(suptitle)
  plt.xlabel('Time')
  plt.ylabel('Count')
  plt.legend(title='Type')
  plt.xticks(rotation=45)
  plt.tight_layout()

  plt.show()

#%%
# Count occurrences of each type per time
df_extrema_lhd = df_extrema[~df_extrema.type.isin(['o', 'c'])].copy()

plot_type_counts(df_extrema_lh, 'All Lows/Highs')

#%%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 14), tight_layout=True)
df_extrema_lh[df_extrema_lh.is_first_of_day].groupby(['time', 'type']).size().unstack(fill_value=0).plot(kind='bar', ax=ax1)
df_extrema_lh[~df_extrema_lh.is_first_of_day].groupby(['time', 'type']).size().unstack(fill_value=0).plot(kind='bar', ax=ax2)
df_extrema_d.groupby(['time', 'type']).size().unstack(fill_value=0).plot(kind='bar', ax=ax3)

for ax in [ax1, ax2, ax3]:
  ax.set_xlabel('Time')
  ax.set_ylabel('Count')
  ax.legend(title='Type')
  ax.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.title('Frequency of Types Over Time')
ax1.set_title('First of Day')
ax2.set_title('Second of Day')
ax3.set_title('Deviations')
plt.show()
