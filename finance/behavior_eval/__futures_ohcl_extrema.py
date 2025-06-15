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
  directory_evals = f'N:/My Drive/Trading/Strategies/swing_ohcl/{symbol}'
  directory_plots = f'N:/My Drive/Trading/Plots/swing_ohcl/{symbol}'
  os.makedirs(directory_evals, exist_ok=True)
  os.makedirs(directory_plots, exist_ok=True)

  # Create a directory
  symbol_def = utils.influx.SYMBOLS[symbol]

  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = dateutil.parser.parse('2020-01-02T00:00:00').replace(tzinfo=tz)
  # first_day = dateutil.parser.parse('2025-03-06T00:00:00').replace(tzinfo=tz)
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

    ##%%
    day_end = day_start + exchange['Close'] + timedelta(hours=1)
    # get the following data for daily assignment
    ##%%

    df_5m = df_5m_two_weeks[(df_5m_two_weeks.index >= day_start+exchange['Open']-timedelta(hours=1, minutes=0)) & (df_5m_two_weeks.index <= day_end)].copy()
    df_30m = df_30m_two_weeks[(df_30m_two_weeks.index >= day_start+exchange['Open']-timedelta(hours=1, minutes=0)) & (df_30m_two_weeks.index <= day_end)].copy()
    df_day = df_day_two_weeks[(df_day_two_weeks.index >= day_start - timedelta(days=14)) & (df_day_two_weeks.index <= day_start + timedelta(days=14))]

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
    cdh = df_5m.h.max()
    cdl = df_5m.l.min()
    cdo = df_5m.o.iat[0]
    cdc = df_5m.c.iloc[-1]
    pdh = prior_day_candle.h.max() if not prior_day_candle.empty else np.nan
    pdl = prior_day_candle.l.min() if not prior_day_candle.empty else np.nan
    pdc = prior_day_candle.c.iat[-1] if not prior_day_candle.empty else np.nan
    cwh = current_week_candle.h.max() if not current_week_candle.empty else np.nan
    cwl = current_week_candle.l.min() if not current_week_candle.empty else np.nan
    pwh = prior_week_candle.h.max()
    pwl = prior_week_candle.l.min()
    onh = overnight_candle.h.max() if not overnight_candle.empty else np.nan
    onl = overnight_candle.l.min() if not overnight_candle.empty else np.nan

    def calculate_offssets(x):
      return {"dpdh": pdh - x, "dpdl": pdl - x, "dcwh": cwh - x, "dcwl": cwl - x, "dpwh": pwh - x, "dpwl": pwl - x, "donh": onh - x, "donl": onl - x}

    start = df_5m.iloc[0]
    low = df_5m['l'].min()
    extreme_low = {"ts": df_5m['l'].idxmin(), "type": 'l', "value": low, **calculate_offssets(low)}
    high = df_5m['h'].max()
    extreme_high = {"ts": df_5m['h'].idxmax(), "type": 'h', "value":high, **calculate_offssets(high)}
    end = df_5m.iloc[-1]

    is_up_move_day = extreme_low['ts'] < extreme_high['ts']
    extrema = [extreme_low, extreme_high] if is_up_move_day else [extreme_high, extreme_low]

    is_open_extreme = extrema[0]['ts'] == start.name
    is_close_extreme = extrema[-1]['ts'] == end.name


    if not is_open_extreme:
      extrema.insert(0,{"ts": df_5m.index[0], "type": 'o', "value":cdo, **calculate_offssets(cdo)})
      
    if not is_close_extreme:
      extrema.append({"ts": df_5m.index[-1], "type": 'c', "value":cdc, **calculate_offssets(cdc)})

    df_extrema = pd.DataFrame(extrema)
    ##%%
    # prior_day = prior_day + timedelta(days=-1)
    # day_start = day_start + timedelta(days=-1)


    ##%%
    # go thought all start < points < end and check distance to the nearest trendline
    is_new_extrema = True
    while is_new_extrema:
      #%%
      for i in range(1, len(df_extrema)):
        ##%%
        day_range = cdh - cdl
        is_new_extrema = False
        ##%%
        start_extrema = df_extrema.iloc[i-1]
        end_extrema = df_extrema.iloc[i]

        df_dir = df_5m[(df_5m.index > start_extrema['ts']) & (df_5m.index < end_extrema['ts'])].copy()
        # if df_dir.empty:
          # continue
        df_dir['dist_h'] = utils.geometry.calculate_y_distance([start_extrema['ts'], start_extrema['value']], [end_extrema['ts'], end_extrema['value']], list(zip(df_dir.index, df_dir['h'])))
        df_dir['dist_l'] = utils.geometry.calculate_y_distance([start_extrema['ts'], start_extrema['value']], [end_extrema['ts'], end_extrema['value']], list(zip(df_dir.index, df_dir['l'])))
        df_dir['dist_h'] = df_dir['dist_h'].abs()
        df_dir['dist_l'] = df_dir['dist_l'].abs()
        df_dir['dist'] = df_dir[['dist_h', 'dist_l']].max(axis=1)

        max_dev = df_dir[df_dir['dist'] > 0.4*day_range]
        if not max_dev.empty:
        # for index, row in max_dev.iterrows():
          time_diff = max_dev.index.diff().fillna(pd.Timedelta(seconds=0))
          group = (time_diff > pd.Timedelta('5m')).cumsum()
          dist_agg = max_dev[group == 0].dist.agg(["idxmax", "max"])
          max_candle = max_dev[max_dev.index == dist_agg['idxmax']]
          value = max_candle.h.iat[0] if max_candle.dist_h.iat[0] > max_candle.dist_l.iat[0] else max_candle.l.iat[0]
          dev_dict = {"ts": dist_agg['idxmax'], "type": 'dev', 'value': value, 'dist': max_candle.dist, **calculate_offssets(value)}
          is_new_extrema = True
          df_extrema = pd.concat([df_extrema, pd.DataFrame([dev_dict])]).sort_values(by='ts')
          df_extrema.reset_index(drop=True, inplace=True)
          break


      # if not is_new_extrema:
      #   continue
      # df_dev = pd.DataFrame(deviations)
      # time_diff = df_dev.ts.diff().fillna(pd.Timedelta(seconds=0))
      # group = (time_diff > pd.Timedelta('5m')).cumsum()
      #
      # ##%%
      # # Aggregate rows by group (example: summing values)
      # df_dev_agg = df_dev.groupby(group).agg(
      #   dist=("dist", "max"),  # You can use other aggregation functions like 'mean', 'max', etc.
      #   ts=("dist", lambda x: df_dev.iloc[x.idxmax()].ts),
      # )
      # df_dev_max = df_dev[df_dev.ts.isin(df_dev_agg.ts)]

    ##%%
    # Filter wrongly selected dev extrema
    is_obsolete = True
    while is_obsolete and any(df_extrema.type == 'dev'):
      for i in range(2, len(df_extrema)):
        start_extrema = df_extrema.iloc[i-2]
        current_extrema = df_extrema.iloc[i-1]
        end_extrema = df_extrema.iloc[i]
        if current_extrema.type != 'dev':
          continue
        is_obsolete = start_extrema.value > current_extrema.value > end_extrema.value or start_extrema.value < current_extrema.value < end_extrema.value
        if is_obsolete:
          df_extrema.drop(i-1, inplace=True)
          df_extrema.reset_index(drop=True, inplace=True)
          break


    # Follow algorithm
    # 1. Start with a candle range defined as [start end] whereas start['l'] is the lowest value and end['h'] is the highest value
    # 2. Count every succession of negative candles

    ##%%
    # pullback = {"start": , "end": , "duration": , "h": , "l": , "o": , "c": , "type": 'SC', "direction": 1 | -1"}"
    ##%%
    pullbacks = []
    for i in range(1, len(df_extrema)):
      start_extrema = df_extrema.iloc[i-1]
      end_extrema = df_extrema.iloc[i]
      direction = 1 if start_extrema.value < end_extrema.value else -1
      df_dir = df_5m[(df_5m.index > start_extrema.ts) & (df_5m.index < end_extrema['ts'])]
      current_pullback = None
      j_start = 0
      for j in range(1, len(df_dir)):
        if direction > 0:
          is_opposite = df_dir.iloc[j]['o'] >= df_dir.iloc[j]['c']
        else:
          is_opposite = df_dir.iloc[j]['o'] <= df_dir.iloc[j]['c']
        if not is_opposite and current_pullback is not None:
          current_pullback['end'] = df_dir.iloc[j-1].name
          current_pullback['l'] = df_dir.iloc[j_start:j].l.min()
          current_pullback['h'] = df_dir.iloc[j_start:j].h.max()
          current_pullback['c'] = df_dir.iloc[j-1]['c']

          pullbacks.append(current_pullback)
          current_pullback = None

        is_positive_candle = df_dir.iloc[j]['o'] <= df_dir.iloc[j]['c']
        # if is_continuation and ((direction > 0 and not is_positive_candle) or (direction < 0 and is_positive_candle)):
        #   pullbacks.append({"start": df_dir.iloc[j-1].name, "end": df_dir.iloc[j].name, "h": df_dir.iloc[j]['h'],
        #                     "l": df_dir.iloc[j]['l'], "o": df_dir.iloc[j]['o'], "c": df_dir.iloc[j]['c'], "type": 'SC', "direction": direction})

        if is_opposite and current_pullback is None:
          current_pullback = {"start": df_dir.iloc[j].name, "o": df_dir.iloc[j]['o'], "direction": direction}
          j_start = j

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
      fig.suptitle(f'{symbol} {date_str} 5m O {cdo:.2f} H {cdh:.2f} C {cdc:.2f} L {cdl:.2f} \n' +
             f'PriorDay: H {pdh:.2f} C {pdc:.2f} L {pdl:.2f}  On: H {onh:.2f} L {onl:.2f} \n' +
             f'CurrentWeek: H {cwh:.2f} L {cwl:.2f}  PriorWeek: H {pwh:.2f} L {pwl:.2f}')

      hlines=dict(hlines=indicator_hlines, colors=[*['#bf42f5']*5, *['#3179f5']*4], linewidths=[0.6]*3+[0.4]*6, linestyle=['--', *['-']*(len(indicator_hlines)-1)])

      ind_5m_ema20_plot = mpf.make_addplot(df_5m['20EMA'], ax=ax1, width=0.6, color="#FF9900", linestyle='--')
      ind_5m_ema240_plot = mpf.make_addplot(df_5m['240EMA'], ax=ax1, width=0.6, color='#0099FF', linestyle='--')
      ind_vwap3_plot = mpf.make_addplot(df_5m['VWAP3'], ax=ax1, width=0.4, color='magenta')

      ind_30m_ema20_plot = mpf.make_addplot(df_30m['20EMA'], ax=ax3, width=0.6, color="#FF9900", linestyle='--')
      ind_30m_ema40_plot = mpf.make_addplot(df_30m['40EMA'], ax=ax3, width=0.6, color='#0099FF', linestyle='--')

      ind_day_ema20_plot = mpf.make_addplot(df_day['20EMA'], ax=ax2, width=0.6, color="#FF9900", linestyle='--')

      df_ohcl_extrema = df_extrema[df_extrema.type.isin(['o', 'h', 'c', 'l'])]
      extrema_ohcl_aline = list(zip(df_ohcl_extrema.ts, df_ohcl_extrema.value))
      extrema_aline = list(zip(df_extrema.ts, df_extrema.value))

      pb_alines= []
      for pullback in pullbacks:
        pb_alines.append([(pullback['start'], pullback['l']), (pullback['end'], pullback['l'])])
        pb_alines.append([(pullback['start'], pullback['h']), (pullback['end'], pullback['h'])])

      alines=dict(alines=[extrema_ohcl_aline, extrema_aline, *pb_alines], colors=['purple', 'black', *['darkblue'] * len(pb_alines)], alpha=0.3, linewidths=[0.25], linestyle=['--']*2+['-'] * len(pb_alines))

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
      ## %%
      metadata = {"pullbacks": pullbacks, "extrema": df_extrema, "VWAP": df_5m['VWAP3']}
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
