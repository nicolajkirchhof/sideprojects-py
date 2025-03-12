# %%
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
from matplotlib.pyplot import tight_layout

from finance.utils.pct import percentage_change
import finplot as fplt
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

from finance.behavior_eval.influx_utils import get_candles_range_aggregate_query, get_candles_range_raw_query
pd.options.plotting.backend = "matplotlib"


mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2

# %% get influx data
DB_INDEX = 'index'
DB_CFD = 'cfd'
DB_FOREX = 'forex'

influx_client_df = idb.DataFrameClient()
influx_client = idb.InfluxDBClient()

indices = influx_client.query('show measurements', database=DB_INDEX)
cfds = influx_client.query('show measurements', database=DB_CFD)
forex = influx_client.query('show measurements', database=DB_FOREX)

get_values = lambda x: [y[0] for y in x.raw['series'][0]['values']]
print('Indices: ', get_values(indices))
print('Cfds: ', get_values(cfds))
print('Forex: ', get_values(forex))


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
symbols = [('IBDE40', pytz.timezone('Europe/Berlin')), ('IBGB100', pytz.timezone('Europe/London')),
           *[(x, pytz.timezone('America/New_York')) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]

symbol, tz = symbols[0]
# Create a directory
directory = f'N:/My Drive/Projects/Trading/Research/Plots/thu_fri_mon'
os.makedirs(directory, exist_ok=True)

# for symbol, tz in symbols:

#%%
  dfs_ref_range = []
  dfs_closing = []
  first_year = 2020
  last_year = 2025

  thursdays = get_thursdays(datetime(2020, 1, 1, hour=9, tzinfo=tz),
                            tz.localize(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)))

  ## %%
  thu = thursdays[0]
  results = []

  for thu in thursdays:
    fri = thu + timedelta(days=1)
    sat = fri + timedelta(days=1)
    mon = sat + timedelta(days=2)
    tue = mon + timedelta(days=1)

    thu_influx = influx_client_df.query(get_candles_range_aggregate_query(thu, thu + timedelta(hours=7), symbol), database=DB_CFD)
    fri_influx = influx_client_df.query(get_candles_range_aggregate_query(fri, fri + timedelta(hours=7), symbol), database=DB_CFD)
    mon_influx = influx_client_df.query(get_candles_range_aggregate_query(mon, mon + timedelta(hours=7), symbol), database=DB_CFD)

    ##%%
    if not symbol in thu_influx or not symbol in fri_influx or not symbol in mon_influx:
      print(f'no data for {thu.isoformat()}')
      continue

    thu_candle = thu_influx[symbol].tz_convert(tz)
    fri_candle = fri_influx[symbol].tz_convert(tz)
    mon_candle = mon_influx[symbol].tz_convert(tz)
    thu_dict = thu_candle.rename(columns={'h': 't_h', 'l':'t_l', 'o':'t_o', 'c': 't_c', 'v': 't_v'}).iloc[0].to_dict()
    fri_dict = fri_candle.rename(columns={'h': 'f_h', 'l':'f_l', 'o':'f_o', 'c': 'f_c', 'v': 'f_v'}).iloc[0].to_dict()
    mon_dict = mon_candle.rename(columns={'h': 'm_h', 'l':'m_l', 'o':'m_o', 'c': 'm_c', 'v': 'm_v'}).iloc[0].to_dict()

    result = {'thu': thu, 'fri': fri, 'mon':mon, **thu_dict, **fri_dict, **mon_dict}
    results.append(result)

  results_df = pd.DataFrame(results)
#%%
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
print(f'lower thu high, higher mon high {num_lower_thu_higher_mon_high} pct {num_lower_thu_higher_mon_high/num_higher_thu_high:.2%}')
num_higher_thu_lower_mon_high = results_df[~lower_thu_high & ~higher_mon_high].shape[0]
print(f'lower thu high, lower mon high {num_lower_thu_lower_mon_high} pct {num_lower_thu_lower_mon_high/num_higher_thu_high:.2%}')
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

results_df.to_pickle(f'{directory}/{symbol}_fri_mon.pkl')

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

