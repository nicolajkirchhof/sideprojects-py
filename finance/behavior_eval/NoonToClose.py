# %%
from datetime import datetime, timedelta

import time
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
symbols = [('DAX', pytz.timezone('Europe/Berlin'), DB_INDEX), *[(x, pytz.timezone('America/New_York'), DB_CFD) for x in ['IBUS30', 'IBUS500', 'IBUST100']]]

symbol, tz, db = symbols[1]
# Create a directory
directory = f'N:/My Drive/Projects/Trading/Research/Strategies/NoonToClose'
os.makedirs(directory, exist_ok=True)
results = []

for symbol, tz, db in symbols[1:]:
##%%%
  first_day = tz.localize(dateutil.parser.parse('2020-01-02T00:00:00'))
  # first_day = tz.localize(dateutil.parser.parse('2024-10-24T00:00:00'))
  last_day =  tz.localize(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
  day_start = first_day

  ## %%
  while day_start < last_day:
    ## %%
    noon = (day_start + timedelta(hours=12)).astimezone(pytz.utc)
    close = (day_start + timedelta(hours=14)).astimezone(pytz.utc)

    day_start = day_start + timedelta(days=1)
    if noon.isoweekday() in [6, 7]:
      continue

    noon_influx = influx_client_df.query(get_candles_range_aggregate_query(noon, noon + timedelta(hours=1), symbol), database=db)
    close_influx = influx_client_df.query(get_candles_range_aggregate_query(close, close + timedelta(hours=3), symbol), database=db)

    ##%%
    if not symbol in noon_influx or not symbol in close_influx:
      print(f'no data for {symbol} {noon.isoformat()} {not symbol in noon_influx } {not symbol in close_influx }')
      continue

    noon_candle = noon_influx[symbol].tz_convert(tz)
    close_candle = close_influx[symbol].tz_convert(tz)
    noon_dict = noon_candle.rename(columns={'h': 'n_h', 'l':'n_l', 'o':'n_o', 'c': 'n_c', 'v': 'n_v'}).iloc[0].to_dict()
    close_dict = close_candle.rename(columns={'h': 'c_h', 'l':'c_l', 'o':'c_o', 'c': 'c_c', 'v': 'c_v'}).iloc[0].to_dict()

    result = {'noon': noon, 'close':close,  **noon_dict, **close_dict}
    results.append(result)


##%%
  results_df = pd.DataFrame(results)
  results_df.to_pickle(f'{directory}/{symbol}_fri_mon.pkl')

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

