# %%
from datetime import datetime, timedelta

import dateutil
import numpy as np

import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
from matplotlib.pyplot import tight_layout

from finance import utils
import pytz

pd.options.plotting.backend = "matplotlib"


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

#%%


#%%

# Group data by the date part of the index
# df_grp = df.groupby(df.index.date)

# %%

def get_thursdays(start_date, end_date):
  # Iterate through all days of the year
  current_date = start_date

  # Check if the current day is a Thursday or Friday
  while current_date.weekday() != 3:
    current_date += timedelta(days=1)

  # Create a list for the results
  result = []

  while current_date <= end_date:
    result.append(current_date)
    # Move to the next day
    current_date += timedelta(days=7)

  return result


# %%
symbols = ['DAX', 'ESTX50', 'SPX']
# Create a directory
directory = f'N:/My Drive/Projects/Trading/Research/Plots/thu_fri_mon'
os.makedirs(directory, exist_ok=True)

symbol = symbols[0]

##%%
# for symbol in symbols:
  symbol_def = utils.influx.SYMBOLS[symbol]
  tz = symbol_def['EX']['TZ']

  first_day = tz.localize(dateutil.parser.parse('2020-01-01T00:00:00'))
  last_day = tz.localize(dateutil.parser.parse('2025-03-19T00:00:00'))
  df = utils.influx.get_candles_range_aggregate(first_day, last_day, symbol, '1d')




#%%
for symbol, tz in symbols:
  results_df = pd.read_pickle(f'{directory}/{symbol}_fri_mon.pkl')
  print(f'{symbol}')
  lower_thu_high = (results_df.t_h < results_df.f_h)
  num_lower_thu_high = results_df[lower_thu_high].shape[0]
  print(f'lower thu high {num_lower_thu_high} pct {num_lower_thu_high/results_df.shape[0]:.2%}')
  num_higher_thu_high = results_df[~lower_thu_high].shape[0]
  print(f'higher thu high {num_higher_thu_high} pct {num_higher_thu_high/results_df.shape[0]:.2%}')

  higher_mon_high = (results_df.m_h > results_df.f_c)
  num_higher_mon_high = results_df[higher_mon_high].shape[0]
  print(f'higher mon high {num_higher_mon_high} pct {num_higher_mon_high/results_df.shape[0]:.2%}')
  num_lower_mon_high = results_df[~higher_mon_high].shape[0]
  print(f'lower mon high {num_lower_mon_high} pct {num_lower_mon_high/results_df.shape[0]:.2%}')

  lower_mon_low = (results_df.m_l < results_df.f_c)
  num_lower_mon_low = results_df[lower_mon_low].shape[0]
  print(f'lower mon low {num_lower_mon_low} pct {num_lower_mon_low/results_df.shape[0]:.2%}')
  num_higer_mon_low =results_df[~lower_mon_low].shape[0]
  print(f'higher mon low {num_higer_mon_low} pct {num_higer_mon_low/results_df.shape[0]:.2%}')

  num_lower_thu_higher_mon_high = results_df[lower_thu_high & higher_mon_high].shape[0]
  print(f'lower thu high, higher mon high {num_lower_thu_higher_mon_high} pct {num_lower_thu_higher_mon_high/num_lower_thu_high:.2%}')
  num_lower_thu_lower_mon_high = results_df[lower_thu_high & ~higher_mon_high].shape[0]
  print(f'lower thu high, lower mon high {num_lower_thu_lower_mon_high} pct {num_lower_thu_lower_mon_high/num_lower_thu_high:.2%}')

  num_higher_thu_higher_mon_high = results_df[~lower_thu_high & higher_mon_high].shape[0]
  print(f'higher thu high, higher mon high {num_higher_thu_higher_mon_high} pct {num_higher_thu_higher_mon_high/num_higher_thu_high:.2%}')
  num_higher_thu_lower_mon_high = results_df[~lower_thu_high & ~higher_mon_high].shape[0]
  print(f'higher thu high, lower mon high {num_higher_thu_higher_mon_high} pct {num_higher_thu_higher_mon_high/num_higher_thu_high:.2%}')
# results_df[(results_df.t_h > results_df.f_h) & (results_df.f_c > results_df.m_l)]
# results_df[(results_df.t_h > results_df.f_h) & (results_df.f_c > results_df.m_l)].size
# results_df[(results_df.t_h > results_df.f_h) & (results_df.f_c > results_df.m_l)].shape
# results_df[(results_df.f_c > results_df.m_l)].shape
# results_df[(results_df.t_h < results_df.f_h) & (results_df.f_c > results_df.m_l)].shape

#%%
# Map the custom column names to the required OHLC column names
column_mapping = list({
                        'Open': 'o',  # Map "Open" to our custom "Start" column
                        'High': 'h',  # Map "High" to "Highest"
                        'Low': 'l',  # Map "Low" to "Lowest"
                        'Close': 'c',  # Map "Close" to "Ending"
                        'Volume': 'v'  # Map "Volume" to "Volume_Traded"
                      }.values())
#%%
fig = mpf.figure(style='yahoo', figsize=(20, 12), tight_layout=True)
fig.suptitle(f'{symbol} thu fri mon')

ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

mpf.plot(results_df.set_index('thu'), ax=ax1, type='candle', columns=[f't_{x}' for x in column_mapping], xrotation=0, datetime_format='%y-%m-%d',
         tight_layout=True, scale_width_adjustment=dict(candle=1.35))
mpf.plot(results_df.set_index('fri'), ax=ax2, type='candle', columns=[f'f_{x}' for x in column_mapping], xrotation=0, datetime_format='%y-%m-%d',
         tight_layout=True, scale_width_adjustment=dict(candle=1.35))
mpf.plot(results_df.set_index('mon'), ax=ax3, type='candle', columns=[f'm_{x}' for x in column_mapping], xrotation=0, datetime_format='%y-%m-%d',
         tight_layout=True, scale_width_adjustment=dict(candle=1.35))
# plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
# plt.savefig(f'finance/_data/dax_plots_mpl/IBDE40_{date_str}.png', dpi=300, bbox_inches='tight')  # High-quality save
plt.show()
# plt.close()
# print(f'finished {symbol} {date_str}')
#%%


#%%
lower_high_triggered = thu_candle.h.iat[0] > fri_candle.h.iat[0]

result = {'thu': thu, 'fri': fri, 'lower_high_triggered': lower_high_triggered,
          'lower_high_success': False, 't_high': thu_candle.h.iat[0],
          'f_high': fri_candle.h.iat[0], 't_low': thu_candle.l.iat[0], 'f_low': fri_candle.l.iat[0],
          'm_high': np.NAN, 'm_low': np.NAN}
if lower_high_triggered:
  if not symbol in mon_influx:
    print(f'no data for monday {mon.isoformat()}')
    continue
  result['lower_high_success'] = mon_candle.l.iat[0] < fri_candle.l.iat[0]
  result['m_high'] =mon_candle.h.iat[0]
  result['m_low'] =mon_candle.l.iat[0]

results.append(result)
##%%
df_results = pd.DataFrame(results)
df_results.to_csv(f'{directory}/{symbol}.csv')

num_triggered = df_results.lower_high_triggered.sum()
num_events = df_results.lower_high_triggered.count()
num_triggered_with_success = (df_results.lower_high_triggered & df_results.lower_high_success).sum()
num_triggered_no_success = (df_results.lower_high_triggered & ~df_results.lower_high_success).sum()
print(f'Symbol {symbol} results:')
print(f'Lower high on Friday triggered {num_triggered} times out of {num_events} events ({num_triggered/num_events:.2%})')
print(f'Lower low on Monday success {num_triggered_with_success} times out of {num_triggered} events ({num_triggered_with_success/num_triggered:.2%})')


  # #%%
  # # df_results.plot(x=['fri', 'fri', 'fri', 'fri'], y=['t_high', 'f_high', 'f_low', 'm_low'], kind='scatter')
  # ax = df_results_triggerd.plot(x='fri', y='t_high', c='blue',kind='scatter')
  # df_results_triggerd.plot(ax=ax, x='fri', y='f_high', c='green',kind='scatter')
  # df_results_triggerd.plot(ax=ax, x='fri', y='f_low', c='black',kind='scatter')
  # df_results_triggerd.plot(ax=ax, x='fri', y='m_low', c='red',kind='scatter')
  #
  # plt.show()
  ##%%
  fig, ax = plt.subplots(figsize=(24, 12), tight_layout=True)
  fig.suptitle(f'{symbol} Lower High Triggered')

  df_results_triggerd = df_results[df_results.lower_high_triggered]
  df_results_triggerd.plot(x='fri', y='f_low', c='black',kind='scatter', ax=ax)
  df_results_triggerd[df_results_triggerd.lower_high_success].plot(ax=ax, x='fri', y='m_low', c='green',kind='scatter')
  df_results_triggerd[~df_results_triggerd.lower_high_success].plot(ax=ax, x='fri', y='m_low', c='red',kind='scatter')

  # Format the major and minor ticks for the x-axis
  ax.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks for months
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format months as "Jan 2023"

  ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for days
  ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))  # Format days as just the day number

  # Rotate date labels for better readability
  plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
  plt.setp(ax.xaxis.get_minorticklabels(), rotation=45)
  ax.grid(visible=True, which="both", linestyle="--", alpha=0.5)  # Grid for major and minor ticks
  plt.savefig(f'{directory}/{symbol}.png', bbox_inches='tight')
  plt.show()

