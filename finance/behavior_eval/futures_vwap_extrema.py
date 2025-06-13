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

import finance.utils as utils
from finance.behavior_eval.trading_setup_plots import day_data

mpl.use('TkAgg')
mpl.use('QtAgg')
#%%
%load_ext autoreload
%autoreload 2


#%%
symbols = ['IBDE40', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225', 'USGOLD' ]
symbol = symbols[0]
for symbol in symbols:
  #%% Create a directory
  ad = True
  ad_str = '_ad' if ad else ''
  directory_evals = f'N:/My Drive/Trading/Strategies/swing_vwap{ad_str}/{symbol}'
  directory_plots = f'N:/My Drive/Trading/Plots/swing_vwap{ad_str}/{symbol}'
  os.makedirs(directory_evals, exist_ok=True)
  os.makedirs(directory_plots, exist_ok=True)

  symbol_def = utils.influx.SYMBOLS[symbol]
  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = tz.localize(dateutil.parser.parse('2020-01-02T00:00:00'))
  # first_day = tz.localize(dateutil.parser.parse('2025-03-06T00:00:00'))
  now = datetime.now(tz)
  last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  day_data = utils.trading_day_data.TradingDayData(symbol)
  #%%
  while day_start < last_day:
    #%%
    day_data.update(day_start, prior_day)

    if not day_data.has_sufficient_data():
      print(f'Skipping day {day_start.date()} because of insufficient data')
      day_start = day_start + timedelta(days=1)
      # continue
      raise(Exception('Insufficient data'))

    df_5m = day_data.df_5m_ad if ad else day_data.df_5m

    # Extrema algorithm
    # 1. Start with the o, h, c, l of the current day
    # 2. Determine if time_h is before or after time_l
    # 3. Assume time_h > time_l for the following steps
    # 4. Determine if time_h == time_o as isOpenHigh
    # 5. Determine if time_l == time_c as isCloseLow
    # 6. If not isOpenHigh
    # 7.
    # extrema = {"ts": , "type": 'h' | 'l',  'dpdh': , 'dpdl' , 'dcwh' , 'dcwl' , 'dpwh' , 'dpwl' }

    def calculate_offssets(x):
      return {"dpdh": day_data.pdh - x, "dpdl": day_data.pdl - x, "dcwh": day_data.cwh - x, "dcwl": day_data.cwl - x,
              "dpwh": day_data.pwh - x, "dpwl": day_data.pwl - x, "donh": day_data.onh - x, "donl": day_data.onl - x}

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
      extrema.insert(0,{"ts": df_5m.index[0], "type": 'o',
                        "value":df_5m.o.iloc[0], **calculate_offssets(df_5m.o.iloc[0])})
      
    if not is_close_extreme:
      extrema.append({"ts": df_5m.index[-1], "type": 'c',
                      "value":df_5m.c.iloc[-1], **calculate_offssets(df_5m.c.iloc[-1])})

    df_extrema = pd.DataFrame(extrema)
    ##%%
    vwap_tops_filter = ((df_5m['VWAP3'].shift(1) < df_5m['VWAP3']) &
                        (df_5m['VWAP3'] > df_5m['VWAP3'].shift(-1)))
    vwap_bottoms_filter = ((df_5m['VWAP3'].shift(1) > df_5m['VWAP3']) &
                           (df_5m['VWAP3'] < df_5m['VWAP3'].shift(-1)))
    df_5m['VWAP3_is_top'] = vwap_tops_filter
    df_5m['VWAP3_is_bottom'] = vwap_bottoms_filter
    vwap_tops_index = df_5m[vwap_tops_filter].index.tolist()
    vwap_bottoms_index = df_5m[vwap_bottoms_filter].index.tolist()
    ##%%
    extrema_vwap = []
    # PULLBACK_THRESHOLD_MULTIPLIER = 0.33
    # PULLBACK_THRESHOLD_MULTIPLIER = 0.2
    ##%%
    for pullback_threshold_multiplier in [0.3]:
      pullback_threshold = pullback_threshold_multiplier*(day_data.cdh - day_data.cdl)
      key = f'dev{pullback_threshold_multiplier*100:n}'
      # print(key)
      for i in range(1, len(df_extrema)):
        ##%%
        start_extrema = df_extrema.iloc[i-1]
        end_extrema = df_extrema.iloc[i]
        direction = 1 if end_extrema.type == 'h' or start_extrema.type == 'l'  else -1
        extrema_range_filter = ((df_5m.index >= start_extrema['ts']) &
                                (df_5m.index < end_extrema['ts']))
        df_dir = df_5m[extrema_range_filter]
        if df_dir.empty:
          continue

        last_pivot = df_dir.iloc[0]
        current_pivot = None
        extrema_candidate = None

        # print(f'Start extrema {start_extrema.ts}, End extrema {end_extrema.ts}, Direction {direction}, LastPivot {last_pivot.VWAP3}', )
        ##%%
        for j in range(1, len(df_dir)):
        ##%%
          # print('ts ', df_dir.iloc[j].name)
          is_pivot = df_dir.iloc[j]['VWAP3_is_top'] | df_dir.iloc[j]['VWAP3_is_bottom']
          if not is_pivot:
            # raise Exception('Not pivot')
            continue
          is_candidate = (((df_dir.iloc[j]['VWAP3_is_top'] and direction < 0) or (df_dir.iloc[j]['VWAP3_is_bottom'] and direction > 0)) and
                          abs(df_dir.iloc[j].VWAP3 - last_pivot.VWAP3) > pullback_threshold)
          # print('Is candidate:', is_candidate)
          if is_candidate:
            is_new = False
            if extrema_candidate is None:
              # Create pullback if last pivot distance is more than PULLBACK_THRESHOLD day range
              extrema_candidate = {"start": last_pivot.name, "o": last_pivot.VWAP3, "type": key, "direction": direction}
              is_new = True

            if is_new or (direction < 0 and extrema_candidate['c'] < df_dir.iloc[j].VWAP3) or (direction > 0 and extrema_candidate['c'] > df_dir.iloc[j].VWAP3):
              # Pullback is actually larger, update the range
              # print('Is higher or new candidate')
              extrema_candidate['end'] = df_dir.iloc[j].name
              extrema_candidate['c'] = df_dir.iloc[j].VWAP3

          # Close pivot on lower pivot bottom / higher pivot top or when the next low / high has a distance of greater than the pullback threshold
          is_extrema_candidate_with_pullback_dist = (extrema_candidate is not None and abs(extrema_candidate['c'] - df_dir.iloc[j].VWAP3) > pullback_threshold)

          # print('Is pullback dist:', is_extrema_candidate_with_pullback_dist)
          is_pivot_move = (df_dir.iloc[j]['VWAP3_is_bottom'] and direction < 0 and (df_dir.iloc[j]['VWAP3'] < last_pivot['VWAP3'] or is_extrema_candidate_with_pullback_dist) or
                           df_dir.iloc[j]['VWAP3_is_top'] and direction > 0 and (df_dir.iloc[j]['VWAP3'] > last_pivot['VWAP3'] or is_extrema_candidate_with_pullback_dist))
          # print(f'Is pivot move:{is_pivot_move}, VWAP {df_dir.iloc[j].VWAP3}')
          if is_pivot_move:
            last_pivot = df_dir.iloc[j]
            if extrema_candidate is not None:
              # print(f'Append Extrema Candidate')
              extrema_vwap.append(extrema_candidate)
              extrema_candidate = None

        if extrema_candidate is not None:
          extrema_vwap.append(extrema_candidate)
    for extrema in extrema_vwap:
      # push start and end of extrema
      extrema_start = {"ts": extrema['start'], "type":extrema['type'] ,'value': extrema['o'],  **calculate_offssets(extrema['o'])}
      extrema_end = {"ts": extrema['end'], "type":extrema['type'], 'value': extrema['c'],  **calculate_offssets(extrema['c'])}
      df_extrema = pd.concat([df_extrema, pd.DataFrame([extrema_start, extrema_end])]).sort_values(by='ts')
      df_extrema.reset_index(drop=True, inplace=True)

    ##%%
    # Follow algorithm
    # 1. Start with a candle range defined as [start end] whereas start['l'] is the lowest value and end['h'] is the highest value
    # 2. Count every succession of negative candles

    ##%%
    # pullback = {"start": , "end": , "duration": , "h": , "l": , "o": , "c": , "type": 'SC', "direction": 1 | -1"}"
    ##%%
    pullbacks = []
    for i in range(1, len(df_extrema[df_extrema.type.isin(['o', 'h', 'c', 'l', 'dev2'])])):
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
      def create_trend(i, type):
        return {"start": df_5m.iloc[i].name, "end": df_5m.iloc[i].name, "o": df_5m.iloc[i][type],
                "c": df_5m.iloc[i][type], "bars": 1, "ema20_o": df_5m.iloc[i]["20EMA"]}

      uptrends = []
      current_uptrend = None
      downtrends = []
      current_downtrend = None
      for i in range(len(df_5m)):
        if current_uptrend is None:
          current_uptrend = create_trend(i, 'l')
        elif df_5m.iloc[i].l > current_uptrend['c']:
          current_uptrend['end'] = df_5m.iloc[i].name
          current_uptrend['c'] = df_5m.iloc[i].l
          current_uptrend['ema20_c'] = df_5m.iloc[i]["20EMA"]
          current_uptrend['bars'] += 1
        else:
          if current_uptrend['end'] != current_uptrend['start']:
            uptrends.append(current_uptrend)
          current_uptrend = create_trend(i, 'l')

        if current_downtrend is None:
          current_downtrend= create_trend(i, 'h')
        elif df_5m.iloc[i].h < current_downtrend['c']:
          current_downtrend['end'] = df_5m.iloc[i].name
          current_downtrend['c'] = df_5m.iloc[i].h
          current_downtrend['ema20_c'] = df_5m.iloc[i]["20EMA"]
          current_downtrend['bars'] += 1
        else:
          if current_downtrend['end'] != current_downtrend['start']:
            downtrends.append(current_downtrend)
          current_downtrend= create_trend(i, 'h')

      df_uptrends = pd.DataFrame(uptrends)
      df_uptrends['trend_support'] = df_uptrends['ema20_o'] < df_uptrends['ema20_c']
      df_downtrends = pd.DataFrame(downtrends)
      if not df_downtrends.empty:
        df_downtrends['trend_support'] = df_downtrends['ema20_o'] > df_downtrends['ema20_c']
##%% Candle pattern algorithms
    # - Oii & iii pattern recognition for L-H and O-C
    # - Doji recognition
    # - High candle range change recognition for L-H and O-C
    df_5m.loc[df_5m['lh'] == 0, 'lh'] = 0.001
    df_5m.loc[df_5m['oc'] == 0, 'oc'] = 0.001
    df_5m['is_doji'] = df_5m.lh/2>df_5m.oc.abs()
    mean_atr_lh = df_5m.lh.mean()
    mean_atr_oc = df_5m.oc.abs().mean()

    df_5m['is_oii'] = ((df_5m.shift(2).h > df_5m.shift(1).h) &
                             (df_5m.shift(2).h > df_5m.h) &
                                (df_5m.shift(2).l < df_5m.shift(1).l) &
                                (df_5m.shift(2).l < df_5m.l))


    df_5m['is_high_lh'] = df_5m.lh/2>mean_atr_lh
    df_5m['is_high_oc'] = df_5m.oc/2>mean_atr_oc

    ts_oii = df_5m[df_5m.is_oii].index.tolist()
    ts_is_high_lh = df_5m[df_5m.is_high_lh].index.tolist()
    ts_is_high_oc = df_5m[df_5m.is_high_oc].index.tolist()


    #%%
    df_ohcl_extrema = df_extrema[df_extrema.type.isin(['o', 'h', 'c', 'l'])]
    extrema_ohcl_aline = list(zip(df_ohcl_extrema.ts, df_ohcl_extrema.value))
    # df_extrema2 = df_extrema[df_extrema.type.isin(['o', 'h', 'c', 'l', 'dev25'])]
    # extrema2_aline = list(zip(df_extrema2.ts, df_extrema2.value))
    df_extrema3 = df_extrema[df_extrema.type.isin(['o', 'h', 'c', 'l', 'dev30'])]
    extrema3_aline = list(zip(df_extrema3.ts, df_extrema3.value))
    # df_extrema4 = df_extrema[df_extrema.type.isin(['o', 'h', 'c', 'l', 'dev4'])]
    # extrema4_aline = list(zip(df_extrema4.ts, df_extrema4.value))

    vlines = dict(vlines=ts_is_high_lh+ts_is_high_oc, colors= ['darkviolet']*len(ts_is_high_lh)+['violet']*len(ts_is_high_oc),
                  linewidths=[0.4], linestyle=['--'])

    pb_alines= []
    for pullback in pullbacks:
      # pb_alines.append([(pullback['start'], pullback['l']), (pullback['end'], pullback['l'])])
      pb_alines.append([(pullback['start'], pullback['h']), (pullback['end'], pullback['h'])])

    ut_alines = []
    for idx, uptrend in df_uptrends[df_uptrends.bars > 2].iterrows():
      ut_alines.append([(uptrend['start'], uptrend['o']), (uptrend['end'], uptrend['c'])])

    dt_alines = []
    if not df_downtrends.empty:
      for idx, downtrend in df_downtrends[df_downtrends.bars > 2].iterrows():
        dt_alines.append([(downtrend['start'], downtrend['o']), (downtrend['end'], downtrend['c'])])

    oii_alines = []
    for ts in ts_oii:
      oii_start = df_5m[df_5m.index < ts].iloc[-2]
      oii_alines.append([(oii_start.name, oii_start.l), (ts, oii_start.l)])
      oii_alines.append([(oii_start.name, oii_start.h), (ts, oii_start.h)])

    alines=dict(alines=[extrema_ohcl_aline, extrema3_aline,  *ut_alines, *dt_alines, *oii_alines],
                colors=['purple', 'cornflowerblue']+
                       ['mediumblue']*len(ut_alines)+['mediumblue']*len(dt_alines) + ['deeppink']*len(oii_alines),
                alpha=0.8, linewidths=[0.6],
                linestyle=['--']*4+['-'] * len(ut_alines+dt_alines+oii_alines))

    utils.plots.daily_change_plot(day_data, alines, f'T {pullback_threshold:.2f}', vlines)

    plt.show()
    # %%
    date_str = day_data.day_start.strftime('%Y-%m-%d')
    metadata = {"pullbacks": pullbacks, "extrema": df_extrema, "VWAP": df_5m['VWAP3'], "uptrends": df_uptrends,
                "downtrends": df_downtrends, "firstBars": df_5m[df_5m.index >= day_data.day_open][0:6],
                "ts_oii": ts_oii, "ts_is_high_lh": ts_is_high_lh, "ts_is_high_oc": ts_is_high_oc}
    with open(f'{directory_evals}/{symbol}_{date_str}.pkl', "wb") as f:
      pickle.dump(metadata, f)
    plt.savefig(f'{directory_plots}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
    plt.close()
    print(f'{symbol} finished {date_str}')
    #%%
    prior_day = day_start
    day_start = day_start + timedelta(days=1)
