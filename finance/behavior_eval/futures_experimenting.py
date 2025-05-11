#%%
import pickle
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
from matplotlib import gridspec
from scipy.spatial import ConvexHull

import finance.utils as utils

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
symbols = ['IBDE40', 'IBES35', 'IBFR40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225']
symbol = symbols[0]
for symbol in symbols:
  #%% Create a directory
  directory_evals = f'N:/My Drive/Projects/Trading/Research/Strategies/swing/{symbol}_swings'
  directory_plots = f'N:/My Drive/Projects/Trading/Research/Plots/swing/{symbol}_swings'
  os.makedirs(directory_evals, exist_ok=True)
  os.makedirs(directory_plots, exist_ok=True)

  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]

  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = tz.localize(dateutil.parser.parse('2020-01-03T00:00:00'))
  # first_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
  now = datetime.now(tz)
  last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  ##%%
  df_5m_two_weeks, df_30m_two_weeks, df_day_two_weeks = utils.influx.get_5m_30m_day_date_range_with_indicators(day_start, day_start+timedelta(days=14), symbol)
  #%%
  while day_start < last_day:
    #%%
    if df_5m_two_weeks.index.max() < day_start + timedelta(days=5):
      df_5m_two_weeks, df_30m_two_weeks, df_day_two_weeks = utils.influx.get_5m_30m_day_date_range_with_indicators(day_start, day_start+timedelta(days=14), symbol)

    #%%
    day_end = day_start + exchange['Close'] + timedelta(hours=1)

    # get the following data for daily assignment
    ##%%

    df_5m = df_5m_two_weeks[(df_5m_two_weeks.index >= day_start+exchange['Open']-timedelta(hours=1, minutes=0)) & (df_5m_two_weeks.index <= day_end)].copy()
    df_30m = df_30m_two_weeks[(df_30m_two_weeks.index >= day_start+exchange['Open']-timedelta(hours=1, minutes=0)) & (df_30m_two_weeks.index <= day_end)].copy()
    df_day = df_day_two_weeks[(df_day_two_weeks.index >= day_start - timedelta(days=14)) & (df_day_two_weeks.index <= day_start + timedelta(days=14))]
#%%

def calculate_price_range_convex_hull(df):
  """
  Calculate convex hull for price range data from DataFrame

  Args:
      df: pandas DataFrame with datetime index and 'h'/'l' columns

  Returns:
      tuple: (hull_dates, hull_values) - Points that form the convex hull
  """
  # Convert dates to numerical values for calculation
  dates_numeric = mdates.date2num(df.index)

  # Create points array combining both high and low prices
  points = np.column_stack((
    np.concatenate([dates_numeric, dates_numeric]),
    np.concatenate([df['h'].values, df['l'].values])
  ))

  # Calculate convex hull
  hull = ConvexHull(points)

  # Get the vertices in order to form a polygon
  # Include the first vertex at the end to close the polygon
  vertices = np.append(hull.vertices, hull.vertices[0])
  hull_points = points[vertices]

  # Convert numeric dates back to datetime
  hull_dates = mdates.num2date(hull_points[:, 0])
  hull_values = hull_points[:, 1]

  return hull_dates, hull_values


#%% Example usage:
# Generate sample data
# Calculate convex hull
hull_dates, hull_values = calculate_price_range_convex_hull(df_5m)

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot high and low values
ax.plot(df_5m.index, df_5m['h'], 'g.', alpha=0.5, label='High')
ax.plot(df_5m.index, df_5m['l'], 'r.', alpha=0.5, label='Low')

# Plot convex hull
ax.plot(hull_dates, hull_values, 'b-', linewidth=2, label='Convex hull')

# Format plot
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Range Convex Hull')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()


#%%
"""
Create a polygon connecting all lows and then all highs in sequence

Args:
    df: pandas DataFrame with datetime index and 'h'/'l' columns
"""
# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create the polygon vertices by going through lows first, then highs in reverse
dates = np.concatenate([
  df_5m.index,                  # forward through dates for lows
  df_5m.index[::-1]            # backward through dates for highs
])

values = np.concatenate([
  df_5m['l'].values,           # all lows in order
  df_5m['h'].values[::-1]      # all highs in reverse order
])

# Plot the price range as a polygon
ax.fill(dates, values, 'lightblue', alpha=0.3, label='Price Range Envelope')
ax.plot(dates, values, 'b-', linewidth=1)

# Plot the original high and low points
ax.plot(df_5m.index, df_5m['h'], 'g.', alpha=0.5, markersize=2, label='High')
ax.plot(df_5m.index, df_5m['l'], 'r.', alpha=0.5, markersize=2, label='Low')

# Format plot
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Range Envelope')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()














  #%%

    if df_5m is None or len(df_5m) < 30:
      day_start = day_start + timedelta(days=1)
      print(f'{symbol} no data for {day_start.isoformat()}')
      # raise Exception('no data')
      continue
    last_saturday = utils.time.get_last_saturday(day_start)
    current_week_candle = df_5m_two_weeks[(df_5m_two_weeks.index >= last_saturday) & (df_5m_two_weeks.index <= prior_day + exchange['Close'])]
    prior_week_candle = df_5m_two_weeks[(df_5m_two_weeks.index >= last_saturday-timedelta(days=7)) & (df_5m_two_weeks.index <= last_saturday)]
    prior_day_candle = df_5m_two_weeks[(df_5m_two_weeks.index >= prior_day + exchange['Open']) & (df_5m_two_weeks.index <= prior_day + exchange['Close'])]
    overnight_candle = df_5m_two_weeks[(df_5m_two_weeks.index >= day_start) & (df_5m_two_weeks.index <= day_start + exchange['Open'] - timedelta(hours=1))]

    # Extrema algorithm
    # 1. Start with the o, h, c, l of the current day
    # 2. Determine if time_h is before or after time_l
    # 3. Assume time_h > time_l for the following steps
    # 4. Determine if time_h == time_o as isOpenHigh
    # 5. Determine if time_l == time_c as isCloseLow
    # 6. If not isOpenHigh
    # 7.
    # extrema = {"ts": , "type": 'h' | 'l',  'dpdh': , 'dpdl' , 'dcwh' , 'dcwl' , 'dpwh' , 'dpwl' }

    ##%%
    pdh = prior_day_candle.h.max()
    pdl = prior_day_candle.l.min()
    pdc = prior_day_candle.c.iat[-1]
    cwh = current_week_candle.h.max() if not current_week_candle.empty else np.nan
    cwl = current_week_candle.l.min() if not current_week_candle.empty else np.nan
    pwh = prior_week_candle.h.max()
    pwl = prior_week_candle.l.min()
    onh = overnight_candle.h.max() if not overnight_candle.empty else np.nan
    onl = overnight_candle.l.min() if not overnight_candle.empty else np.nan

    start = df_5m.iloc[0]
    low = df_5m['l'].min()
    extreme_low = {"ts": df_5m['l'].idxmin(), "type": 'l', "value": low, "dpdh": pdh-low, "dpdl": pdl-low,
                   "dcwh": cwh-low, "dcwl": cwl-low, "dpwh": pwh-low, "dpwl": pwl-low, "donh": onh-low, "donl": onl-low}
    high = df_5m['h'].max()
    extreme_high = {"ts": df_5m['h'].idxmax(), "type": 'h', "value":high, "dpdh": pdh-high, "dpdl": pdl-high,
                    "dcwh": cwh-high, "dcwl": cwl-low, "dpwh": pwh-high, "dpwl": pwl-high, "donh": onh-high, "donl": onl-high}
    end = df_5m.iloc[-1]

    is_up_move_day = extreme_low['ts'] < extreme_high['ts']
    extrema_lh = [extreme_low, extreme_high] if is_up_move_day else [extreme_high, extreme_low]

    is_open_extreme = extrema_lh[0]['ts'] == start.name
    is_close_extreme = extrema_lh[-1]['ts'] == end.name

    ##%%
    def get_extrema(df, is_next_high, begin, stop):
      current_direction = -1 if is_next_high else 1
      current = begin
      extrema = []
      while current < stop if begin < stop else current > stop:
        slice_index = (df.index > current) & (df.index <= stop) if begin < stop else (df.index < current) & (df.index >= stop)
        if not any(slice_index):
          return extrema
        if current_direction < 0:
          next_extrema = df[slice_index]['l'].idxmin()
          value = df[slice_index]['l'].min()
        else:
          next_extrema = df[slice_index]['h'].idxmax()
          value = df[slice_index]['h'].max()

        extrema.append({"ts": next_extrema, "type": 'h' if current_direction > 0 else 'l', "value": value,
                        "dpdh": pdh-value, "dpdl": pdl-value, "dcwh": cwh-value, "dcwl": cwl-value, "dpwh": pwh-value,
                        "dpwl": pwl-value, "donh": onh-value, "donl": onl-value})
        current = next_extrema
        current_direction = -1*current_direction
      return extrema

    ##%%
    start_extrema = [] if is_open_extreme else get_extrema(df_5m, ~is_up_move_day, extrema_lh[0]['ts'], start.name)
    start_extrema.reverse()
    end_extrema = [] if is_close_extreme else get_extrema(df_5m, is_up_move_day, extrema_lh[-1]['ts'], end.name)
    extrema_ordered = [*start_extrema, *extrema_lh, *end_extrema]

    # Follow algorithm
    # 1. Start with a candle range defined as [start end] whereas start['l'] is the lowest value and end['h'] is the highest value
    # 2. Follow the lows of the candles as long as the last candle low C_1l is <= current candle low C_l continue
    # 3. If C_l < C_1l follow the pullback until C_l > C_1l again.
    #    a. Determine the duration of the pullback as well as the difference from the first pullback candle high to the last pullback candle low
    #       and the first pullback candle opens to the last pullback candle close.

    ##%%
    # SC = SingleCandle in the opposite direction
    # PB = Pullback that breaks the continuation of the previous candle following
    # pullback = {"start": , "end": , "duration": , "h": , "l": , "o": , "c": , "type": 'SC', "direction": 1 | -1"}"
    ##%%
    pullbacks = []
    for i in range(1, len(extrema_ordered)):
      start_extrema = extrema_ordered[i-1]
      end_extrema = extrema_ordered[i]
      direction = 1 if start_extrema['type'] == 'l' else -1
      df_dir = df_5m[(df_5m.index > start_extrema['ts']) & (df_5m.index < end_extrema['ts'])]
      current_pullback = None
      for j in range(1, len(df_dir)):
        if direction > 0:
          is_continuation = df_dir.iloc[j-1]['l'] <= df_dir.iloc[j]['l']
        else:
          is_continuation = df_dir.iloc[j-1]['h'] >= df_dir.iloc[j]['h']
        if is_continuation and current_pullback is not None:
          current_pullback['end'] = df_dir.iloc[j-1].name
          if direction > 0:
            current_pullback['l'] = df_dir.iloc[j-1]['l']
          else:
            current_pullback['h'] = df_dir.iloc[j-1]['h']
          current_pullback['c'] = df_dir.iloc[j-1]['c']
          pullbacks.append(current_pullback)
          current_pullback = None

        is_positive_candle = df_dir.iloc[j]['o'] <= df_dir.iloc[j]['c']
        # if is_continuation and ((direction > 0 and not is_positive_candle) or (direction < 0 and is_positive_candle)):
        #   pullbacks.append({"start": df_dir.iloc[j-1].name, "end": df_dir.iloc[j].name, "h": df_dir.iloc[j]['h'],
        #                     "l": df_dir.iloc[j]['l'], "o": df_dir.iloc[j]['o'], "c": df_dir.iloc[j]['c'], "type": 'SC', "direction": direction})

        if not is_continuation and current_pullback is None:
          current_pullback = {"start": df_dir.iloc[j-1].name, "o": df_dir.iloc[j-1]['o'], "type": 'PB', "direction": direction}
          if direction > 0:
            current_pullback['h'] = df_dir.iloc[j-1]['h']
          else:
            current_pullback['l'] = df_dir.iloc[j-1]['l']

  ##%%
    try:
##%%
# New setup
# |-------------------------|
# |           5m            |
# | ------------------------|
# |   D   |       30m       |
# | ------------------------|

      fig = mpf.figure(style='yahoo', figsize=(16,9), tight_layout=True)
      gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 2])


      date_str = day_start.strftime('%Y-%m-%d')
      ax1 = fig.add_subplot(gs[0, :])
      ax2 = fig.add_subplot(gs[1, 0])
      ax3 = fig.add_subplot(gs[1, 1])

      indicator_hlines = [pdc, pdh, pdl, onh, onl, cwh, cwl, pwh, pwl]
      fig.suptitle(f'{symbol} {date_str} 5m O {df_5m.o.iat[0]:.2f} H {df_5m.h.iat[0]:.2f} C {df_5m.c.iat[0]:.2f} L {df_5m.l.iat[0]:.2f} \n' +
             f'PriorDay: H {pdh:.2f} C {pdc:.2f} L {pdl:.2f}  On: H {onh:.2f} L {onl:.2f} \n' +
             f'CurrentWeek: H {cwh:.2f} L {cwl:.2f}  PriorWeek: H {pwh:.2f} L {pwl:.2f}')

      hlines=dict(hlines=indicator_hlines, colors=[*['#bf42f5']*5, *['#3179f5']*4], linewidths=[0.6]*3+[0.4]*6, linestyle=['--', *['-']*(len(indicator_hlines)-1)])

      ind_5m_ema20_plot = mpf.make_addplot(df_5m['20EMA'], ax=ax1, width=0.6, color="#FF9900", linestyle='--')
      ind_5m_ema240_plot = mpf.make_addplot(df_5m['240EMA'], ax=ax1, width=0.6, color='#0099FF', linestyle='--')
      ind_vwap3_plot = mpf.make_addplot(df_5m['VWAP3'], ax=ax1, width=0.4, color='magenta')

      ind_30m_ema20_plot = mpf.make_addplot(df_30m['20EMA'], ax=ax3, width=0.6, color="#FF9900", linestyle='--')
      ind_30m_ema40_plot = mpf.make_addplot(df_30m['40EMA'], ax=ax3, width=0.6, color='#0099FF', linestyle='--')

      ind_day_ema20_plot = mpf.make_addplot(df_day['20EMA'], ax=ax2, width=0.6, color="#FF9900", linestyle='--')

      extrema_aline= []
      for extrema in extrema_ordered:
        extrema_aline.append((extrema['ts'], extrema['value']))

      pb_alines= []
      sc_alines= []
      for pullback in pullbacks:
        if pullback['type'] == 'PB':
          pb_alines.append([(pullback['start'], pullback['l']), (pullback['end'], pullback['l'])])
          pb_alines.append([(pullback['start'], pullback['h']), (pullback['end'], pullback['h'])])
        else:
          sc_alines.append([(pullback['start'], pullback['l']), (pullback['end'], pullback['l'])])
          sc_alines.append([(pullback['start'], pullback['h']), (pullback['end'], pullback['h'])])

      alines=dict(alines=[extrema_aline, *pb_alines, *sc_alines], colors=['black', *['darkblue'] * len(pb_alines), *['darkred'] * len(sc_alines)], alpha=0.3, linewidths=[0.25], linestyle=['--', *['-'] * len(pb_alines)])

      mpf.plot(df_5m, type='candle', ax=ax1, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True,
               scale_width_adjustment=dict(candle=1.35), hlines=hlines, alines=alines, addplot=[ind_5m_ema20_plot, ind_5m_ema240_plot, ind_vwap3_plot])
      mpf.plot(df_day, type='candle', ax=ax2, columns=utils.influx.MPF_COLUMN_MAPPING,  xrotation=0, datetime_format='%m-%d', tight_layout=True,
               hlines=hlines, warn_too_much_data=700, addplot=[ind_day_ema20_plot])
      mpf.plot(df_30m, type='candle', ax=ax3, columns=utils.influx.MPF_COLUMN_MAPPING, xrotation=0, datetime_format='%H:%M', tight_layout=True,
               scale_width_adjustment=dict(candle=1.35), hlines=hlines, addplot=[ind_30m_ema20_plot, ind_30m_ema40_plot])


      # Use MaxNLocator to increase the number of ticks
      ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20))  # Increase number of ticks on x-axis
      ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))  # Increase number of ticks on y-axis

      # plt.show()
      # %%
      metadata = {"pullbacks": pullbacks, "extrema": extrema_ordered, "VWAP": df_5m['VWAP3']}
      with open(f'{directory_evals}/{symbol}_{date_str}.pkl', "wb") as f:
        pickle.dump(metadata, f)
      plt.savefig(f'{directory_plots}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
      plt.close()
      print(f'{symbol} finished {date_str}')
      ##%%
      prior_day = day_start
      day_start = day_start + timedelta(days=1)
    except Exception as e:
      print(f'{symbol} error: {e}')
      continue

