#%%
import datetime

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

from finance.utils.pct import percentage_change
import finplot as fplt
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score

from finance.behavior_eval.influx_utils import get_candles_range_aggregate_query, get_candles_range_raw_query

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%% get influx data
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

#%%
#%%
def create_interactive_plot(ax, interval, df):
  fplt.candlestick_ochl(df[['o', 'c', 'h', 'l']], ax=ax)
  hover_label = fplt.add_legend(interval, ax=ax)
  hover_label.opts['color']='#000'

  # #######################################################
  # ## update crosshair and legend when moving the mouse ##
  #
  # def update_legend_text(x, y):
  #   print(interval)
  #   row = df.loc[pd.to_datetime(x, unit='ns', utc=True)]
  #   # format html with the candle and set legend
  #   fmt = '<span style="font-size:15px;color:#%s;background-color:#fff">%%.2f</span>' % ('0d0' if (row.o<row.c).all() else 'd00')
  #   rawtxt = '<span style="font-size:14px">%%s %%s</span> &nbsp; O %s C %s H %s L %s' % (fmt, fmt, fmt, fmt)
  #   values = [row.o, row.c, row.h, row.l]
  #   hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))

  def update_crosshair_text(x, y, xtext, ytext):
    row =  df.iloc[x]
    ytext = f'{y:.2f} O {row.c:.2f} H {row.h:.2f} L {row.l:.2f} C {row.c:.2f}'
    return xtext, ytext

  # fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
  fplt.add_crosshair_info(update_crosshair_text, ax=ax)

# Function to save screenshot of the chart
def save_screenshot(filename:str):
  # Grab the Finplot window as a QPixmap
  pixmap = fplt.app.activeWindow().grab()
  # Save the pixmap to an image file (e.g., PNG)
  pixmap.save(filename)
  print(f"Screenshot saved as '{filename}'")
  fplt.close()

#%%


#%%
symbol = 'IBDE40'
tz = pytz.timezone('Europe/Berlin')
# Create a directory
directory = f'N:/My Drive/Projects/Trading/Research/Plots/{symbol}_mpf_2m_10m_60m'
os.makedirs(directory, exist_ok=True)

# symbol = 'SPX'
# tz = pytz.timezone('EST')

dfs_ref_range = []
dfs_closing = []
first_day = tz.localize(dateutil.parser.parse('2020-01-02T00:00:00'))
last_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
day_start = first_day
#%%
while day_start < last_day:
  day_end = day_start + datetime.timedelta(days=1)
  # get the following data for daily assignment
  day_candles = influx_client_df.query(get_candles_range_raw_query(day_start, day_end, symbol), database=DB_CFD)
  day_start = day_end
  if symbol not in day_candles:
    print(f'no data for {day_start.isoformat()}')
    continue
##%%
  df_raw = day_candles[symbol].tz_convert(tz)
  df_2m = df_raw.resample('2min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  # df_5m = df_raw.resample('5min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  df_10m = df_raw.resample('10min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  # df_15m = df_raw.resample('15min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  # df_30m = df_raw.resample('30min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
  df_60m = df_raw.resample('60min').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))

  # df_2m.index = df_2m.index + pd.DateOffset(minutes=-1)
  # df_10m.index = df_10m.index + pd.DateOffset(minutes=-1)
  # df_60m.index = df_60m.index + pd.DateOffset(minutes=-1)
  df_10m.index = df_10m.index + pd.DateOffset(minutes=5)
  df_60m.index = df_60m.index + pd.DateOffset(minutes=30)

  # Map the custom column names to the required OHLC column names
  column_mapping = list({
                          'Open': 'o',  # Map "Open" to our custom "Start" column
                          'High': 'h',  # Map "High" to "Highest"
                          'Low': 'l',  # Map "Low" to "Lowest"
                          'Close': 'c',  # Map "Close" to "Ending"
                          'Volume': 'v'  # Map "Volume" to "Volume_Traded"
                        }.values())
##%%
  try:
    fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)

    date_str = day_start.strftime('%Y-%m-%d')
    fig.suptitle(f'{symbol} {date_str} 2m/10m/60m')
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    mpf.plot(df_2m, type='candle', ax=ax1, columns=column_mapping,  xrotation=0, datetime_format='%H:%M', tight_layout=True)
    mpf.plot(df_10m, type='candle', ax=ax2, columns=column_mapping, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35))
    mpf.plot(df_60m, type='candle', ax=ax3, columns=column_mapping, xrotation=0, datetime_format='%H:%M', tight_layout=True, scale_width_adjustment=dict(candle=1.35))
    # ax1.set_title('2m')
    # ax2.set_title('10m')
    # ax3.set_title('60m')
    ticks = pd.date_range(df_2m.index.min(),df_2m.index.max(),freq='30min')
    ticklabels = [ tick.time().strftime('%H:%M') for tick in ticks ]
    loc_2m = [ df_2m.index.get_loc(tick) for tick in ticks ]
    ax1.set_xticks(loc_2m)
    ax1.set_xticklabels(ticklabels)
    ax1.set_xlim(-2.5, loc_2m[-1]+15)

    ticks = pd.date_range(df_2m.index.min(),df_2m.index.max(),freq='30min')+pd.Timedelta(minutes=15)
    ticklabels = [ tick.time().strftime('%H:%M') for tick in ticks ]
    loc_10m = [ df_10m.index.get_loc(tick) for tick in ticks ]
    ax2.set_xticks(loc_10m)
    ax2.set_xticklabels(ticklabels)
    ax2.set_xlim(-1, loc_10m[-1]+1.5)

    ticks = pd.date_range(df_2m.index.min(),df_2m.index.max(),freq='1h')+pd.DateOffset(minutes=30)
    ticklabels = [ tick.time().strftime('%H:%M') for tick in ticks ]
    loc_60m = [ df_60m.index.get_loc(tick) for tick in ticks ]
    ax3.set_xticks([ df_60m.index.get_loc(tick) for tick in ticks ])
    ax3.set_xticklabels(ticklabels)
    ax3.set_xlim(-0.583, loc_60m[-1]+0.5)



    # plt.savefig(f'finance/_data/dax_plots_mpl/IBDE40_{date_str}.png', dpi=300, bbox_inches='tight')  # High-quality save
    plt.savefig(f'{directory}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
    # plt.savefig(f'N:/My Drive/Projects/Trading/Research/DAX/IBDE40_mpf_2m_10m_60m/IBDE40_{date_str}.svg', bbox_inches='tight')  # High-quality save
    # plt.savefig(f'N:/My Drive/Projects/Trading/Research/DAX/IBDE40_mpf_2m_10m_60m/IBDE40_{date_str}.jpg', bbox_inches='tight')  # High-quality save
    plt.close()
    # plt.show()
    print(f'finished {date_str}')
  except Exception as e:
    print(f'error: {e}')
    continue

#%%
# # Create the new row with a time index before the existing ones
# first_row = pd.DataFrame({'o': np.NAN, 'h': np.NAN, 'c': np.NAN, 'l': np.NAN}, index=[df_raw.index.min() - datetime.timedelta(minutes=5)])
# last_row = pd.DataFrame({'o': np.NAN, 'h': np.NAN, 'c': np.NAN, 'l': np.NAN}, index=[df_raw.index.max() + datetime.timedelta(minutes=5)])
#
# # Prepend the new row
# # df_2m = pd.concat([first_row, df_2m, last_row])
# # df_10m = pd.concat([first_row, df_10m, last_row])
# df_60m = pd.concat([first_row, df_60m])
#
# #%%
# ax, ax2, ax3, ax4 = fplt.create_plot('IBDE40', rows=4)
# create_interactive_plot(ax, '1m', df_raw)
# create_interactive_plot(ax2, '5m', df_5m)
# create_interactive_plot(ax3, '10m', df_10m)
# create_interactive_plot(ax4, '15m', df_15m)
#
# date_str = day_start.strftime('%Y-%m-%d')
# # fplt.timer_callback(lambda: save_screenshot(f'finance/_data/dax_plots/IBDE40_{date_str}.png'), 0.5, single_shot=True)  # Save after 500 ms
# fplt.show()
#
# #%%
# class TypeOfTrade:
#   BREAK_OUT = 'break_out'
#   FAKE_OUT = 'fake_out'
#
# ENTRY_OFFSET = 0.1 # ATR pct
# atr = lambda x: x.h - x.l
# long_sl = lambda x: x.h + atr(x) * ENTRY_OFFSET
# short_sl = lambda x: x.l - atr(x) * ENTRY_OFFSET
# #%%
# df = df_5m
# last_row = None
# trades = []
# active_trade = None
# for index, row in df.iterrows():
#   if last_row is None:
#     last_row = row
#     continue
#   if active_trade is None and row.h > long_sl(last_row) and row.l < short_sl(last_row):
#     # Fake out
#     trades.append({'in':row.name, 'out':row.name, 'type':TypeOfTrade.FAKE_OUT})
#   if active_trade is None and row.h < short_sl(last_row) and row.l > long_sl(last_row):
#     # smaller candle that is simply neglected
#     continue
#   if active_trade is not None and row.h > last_row.h and row.l < last_row.l:
#     if row.c > first_row.c:
#       print(f'fake out: {row.name}')
#     else:
#       print(f'break out: {row.name}')
#   first_row = row
#
#
#
#
# #%%
# dfs_diff = []
# df_result = []
# for df_in, df_out in zip(dfs_ref_range, dfs_closing):
#   df_input = (df_in.c - df_in.o) / (df_in.h - df_in.l)
#   if not df_input.isna().any():
#     dfs_diff.append(df_input.reset_index(drop=True))
#     df_result.append(percentage_change(df_out.c.iat[0],df_in.iloc[-1].c))
#
#
# ##%%
# dfs_diff_df = pd.concat(dfs_diff, axis=1).T
# dfs_diff_df_pn = dfs_diff_df.map(lambda x: 1 if x > 0 else 0)
# df_result_logreg = [0 if r < 0 else 1 for r in  df_result]
#
# #%%
# def create_interactive_plot(ax, interval, df):
#   fplt.candlestick_ochl(df[['o', 'c', 'h', 'l']], ax=ax)
#   ax.showGrid(y=True)
#   # fplt.set_x_pos(df_raw.index.min(), df_raw.index.max(), ax=ax)
#   hover_label = fplt.add_legend(interval, ax=ax)
#   hover_label.opts['color']='#000'
#
#   # #######################################################
#   # ## update crosshair and legend when moving the mouse ##
#   #
#   # def update_legend_text(x, y):
#   #   print(interval)
#   #   row = df.loc[pd.to_datetime(x, unit='ns', utc=True)]
#   #   # format html with the candle and set legend
#   #   fmt = '<span style="font-size:15px;color:#%s;background-color:#fff">%%.2f</span>' % ('0d0' if (row.o<row.c).all() else 'd00')
#   #   rawtxt = '<span style="font-size:14px">%%s %%s</span> &nbsp; O %s C %s H %s L %s' % (fmt, fmt, fmt, fmt)
#   #   values = [row.o, row.c, row.h, row.l]
#   #   hover_label.setText(rawtxt % tuple([symbol, interval.upper()] + values))
#
#   def update_crosshair_text(x, y, xtext, ytext):
#     row =  df.iloc[x]
#     ytext = f'{y:.2f} O {row.c:.2f} H {row.h:.2f} L {row.l:.2f} C {row.c:.2f}'
#     return xtext, ytext
#
#   # fplt.set_mouse_callback(update_legend_text, ax=ax, when='hover')
#   fplt.add_crosshair_info(update_crosshair_text, ax=ax)
# #%%
# fplt.right_margin_candles = 0
# fplt.side_margin = 0
# # fplt.winx = 0
# # fplt.winy = 0
# # fplt.winh = 2160
# # fplt.winw = 3840
# fplt.timestamp_format = '%H:%M'
# fplt.time_splits = [('years', 2*365*24*60*60,    'YS',  4), ('months', 3*30*24*60*60,   'MS', 10), ('weeks',   3*7*24*60*60, 'W-MON', 10),
#                ('days',      3*24*60*60,     'D', 10), ('hours',        9*60*60,   'h', 16), ('hours',        3*60*60,     'h', 16),
#                ('minutes',        45*60, '15min', 16), ('minutes',        15*60, '5min', 16), ('minutes',         3*60,   'min', 16),
#                ('seconds',           45,   '15s', 19), ('seconds',           15,   '5s', 19), ('seconds',            3,     's', 19),
#                ('milliseconds',       0,    'ms', 23)]
# #%%
# ax, ax2, ax3 = fplt.create_plot('DAX', rows=3)
# # ax.decouple()
# # ax2.decouple()
# # ax3.decouple()
#
# create_interactive_plot(ax, '2m', df_2m)
# # create_interactive_plot(ax2, '5m', df_5m)
# create_interactive_plot(ax2, '10m', df_10m)
#
# # create_interactive_plot(ax4, '15m', df_15m)
# create_interactive_plot(ax3, '60m', df_60m)
# # create_interactive_plot(ax3, '50m', df_50m)
# fplt.windows[0].ci.layout.setRowStretchFactor(0, 1)
# fplt.windows[0].ci.layout.setRowStretchFactor(1, 1)
# fplt.windows[0].ci.layout.setRowStretchFactor(2, 1)
#
# fplt.show()
#
# #%%
# # ax, ax2, ax3 = fplt.create_plot('DAX', rows=3)
# fplt.create_plot('DAX')
# fplt.candlestick_ochl(df_60m[['o', 'c', 'h', 'l']], candle_width=30)
# fplt.candlestick_ochl(df_10m[['o', 'c', 'h', 'l']], candle_width=5)
# fplt.candlestick_ochl(df_2m[['o', 'c', 'h', 'l']])
# fplt.show()
#
# #%%
# # no margins
# fplt.right_margin_candles = 0
# fplt.side_margin = 0
# ax= fplt.create_plot('DAX', rows=1)
# fplt.candlestick_ochl(df[['o', 'c', 'h', 'l']], ax=ax)
# # Aligning ticks to each hour
# # ax_date_range = pd.date_range(df.index.min(), df.index.max(), freq='h')
# # num_ticks = int((df.index.max()- df.index.min()).total_seconds()/3600)
# # ax_date_range = ax.getAxis('bottom').tickValues(df.index.min(), df.index.max(), num_ticks)
# # ax.getAxis('bottom').setTicks([
# # ax.getAxis('bottom').setTicks([
# #   [(date.timestamp(), date.strftime('%H:%M')) for date in ax_date_range]
# # ])
# fplt.show()
#
# #%%
#
