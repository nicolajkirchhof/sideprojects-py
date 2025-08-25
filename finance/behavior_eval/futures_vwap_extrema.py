#%%
import pickle
from datetime import datetime, timedelta
from glob import glob
from zoneinfo import ZoneInfo

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

mpl.use('TkAgg')
mpl.use('QtAgg')
%load_ext autoreload
%autoreload 2


#%%
symbols = ['IBDE40', 'IBEU50', 'IBES35', 'IBGB100', 'IBUS30', 'IBUS500', 'IBUST100', 'IBJP225', 'USGOLD' ]
symbol = symbols[0]
IS_PLOT_ACTIVE=True
IS_ALL_DAY=True
RECREATE=True

for symbol in symbols:
  #%% Create a directory
  ad_str = '_ad' if IS_ALL_DAY else ''
  directory_evals = f'N:/My Drive/Trading/Strategies/swing_vwap{ad_str}/{symbol}'
  directory_plots = f'N:/My Drive/Trading/Plots/swing_vwap{ad_str}/{symbol}'
  os.makedirs(directory_evals, exist_ok=True)
  os.makedirs(directory_plots, exist_ok=True)

  symbol_def = utils.influx.SYMBOLS[symbol]
  exchange = symbol_def['EX']
  tz = exchange['TZ']

  dfs_ref_range = []
  dfs_closing = []
  first_day = dateutil.parser.parse('2020-01-02T00:00:00').replace(tzinfo=tz)

  if not RECREATE:
    basenames = [os.path.basename(f) for f in glob(f'{directory_evals}/*.pkl')]
    date_strs = [base_name.split('_')[-1].replace('.pkl', '') for base_name in basenames]
    dates = [datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=tz) for date_str in date_strs]
    first_day = max(dates) if dates else first_day

  # first_day = dateutil.parser.parse('2025-03-06T00:00:00').replace(tzinfo=tz)
  now = datetime.now(tz)
  last_day = datetime(now.year, now.month, now.day, tzinfo=tz)

  prior_day = first_day
  day_start = first_day + timedelta(days=1)

  day_data = utils.trading_day_data.TradingDayData(symbol, min_future_cache=timedelta(days=365))
  #%%
  while day_start < last_day:
    for df_candles in [day_data.df_5m, day_data.df_10m, day_data.df_15m]:
      #%%
      day_data.update(day_start, prior_day)

      if not day_data.has_sufficient_data():
        print(f'Skipping day {day_start.date()} because of insufficient data')
        day_start = day_start + timedelta(days=1)
        # continue
        raise(Exception('Insufficient data'))

      df_5m = day_data.df_5m_ad if IS_ALL_DAY else day_data.df_5m
      df_10m = day_data.df_10m_ad if IS_ALL_DAY else day_data.df_10m
      df_15m = day_data.df_15m_ad if IS_ALL_DAY else day_data.df_15m
      df_extrema, pullback_threshold = utils.indicators.trading_day_moves(day_data)

      pullbacks, df_uptrends, df_downtrends = utils.indicators.trends(df_5m, df_extrema)

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
      def cancel_moves_stats(candle, next_extrema):
        ##%%
        candle_sentiment = 1 if candle.c - candle.o >= 0 else -1
        candle_atr_pts = candle.h - candle.l
        move_sentiment = 1 if next_extrema.value - candle['VWAP3'] >=0 else -1
        in_trend = candle_sentiment == move_sentiment
        pts_move = abs(candle['VWAP3'] - next_extrema.value)
        candles_move = df_5m[(df_5m.index > candle.name) & (df_5m.index <= next_extrema.ts)]
        bull_trigger = candles_move[candles_move.h > candle.h]
        bear_trigger = candles_move[candles_move.l < candle.l]

        if bull_trigger.empty and bear_trigger.empty:
          bracket_trigger = 0
        elif bull_trigger.empty and not bear_trigger.empty:
          bracket_trigger = -1
        elif not bull_trigger.empty and bear_trigger.empty:
          bracket_trigger = 1
        elif not bull_trigger.empty and not bear_trigger.empty and bull_trigger.index[0] > bear_trigger.index[0]:
          bracket_trigger = -1
        elif not bull_trigger.empty and not bear_trigger.empty and bull_trigger.index[0] < bear_trigger.index[0]:
          bracket_trigger = 1
        else: # same candle so you have to assume you trigger in the wrong direction
          bracket_trigger = move_sentiment * -1

        bracket_candle_move = candles_move[candles_move.index >= bull_trigger.index[0]] if bracket_trigger > 0 else candles_move[candles_move.index >= bear_trigger.index[0]]  if bracket_trigger < 0 else None
        bracket_sl_value = np.nan if bracket_candle_move is None else bracket_candle_move.l.min() if move_sentiment > 0 else bracket_candle_move.h.max()

        sl_value = candles_move.l.min() if move_sentiment > 0 else candles_move.h.max()
        sl_pts_offset = candle.l - sl_value if move_sentiment > 0 else  sl_value - candle.h
        result = {"ts": candle.name, "candle_sentiment": candle_sentiment, "candle_atr_pts": candle_atr_pts,
                "in_trend": in_trend,  "pts_move": pts_move, "sl_value": sl_value, "sl_pts_offset": sl_pts_offset,
                "bracket_trigger": bracket_trigger, "bracket_sl_value": bracket_sl_value}
        ##%%
        return result

      def get_bracket_moves(df, offset):
        ##%%
        bracket_moves = []
        bracket_moves_after_next = []
        for i in range(len(df)-offset):
          ##%%
          candle = df.iloc[i]
          next_extremas = df_extrema[df_extrema.ts > candle.name]
          next_extrema = next_extremas.iloc[0]
          bracket_moves.append(cancel_moves_stats(candle, next_extrema))

          candle = df.iloc[i]
          after_next_extrema = next_extremas.iloc[1] if len(next_extremas) > 1 else None
          if after_next_extrema is not None:
            bracket_moves_after_next.append(cancel_moves_stats(candle, after_next_extrema))

        df_bracket_moves = pd.DataFrame(bracket_moves)
        df_bracket_moves_after_next = pd.DataFrame(bracket_moves_after_next)
        return df_bracket_moves, df_bracket_moves_after_next

      df_bracket_moves, df_bracket_moves_after_next = get_bracket_moves(df_5m, 10)
      df_bracket_moves_10m, df_bracket_moves_after_next_10m = get_bracket_moves(df_10m, 5)
      df_bracket_moves_15m, df_bracket_moves_after_next_15m = get_bracket_moves(df_15m, 3)

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

      date_str = day_data.day_start.strftime('%Y-%m-%d')
      if IS_PLOT_ACTIVE:
        #%%
        utils.plots.daily_change_plot(day_data, alines, f'T {pullback_threshold:.2f}', vlines)

        utils.plots.daily_change_plot(day_data, alines, f'T {pullback_threshold:.2f}', vlines, basetime='10m')

        plt.show()
        #%%
        plt.savefig(f'{directory_plots}/{symbol}_{date_str}.png', bbox_inches='tight')  # High-quality save
        plt.close()
      # %%
      metadata = {"pullbacks": pullbacks, "extrema": df_extrema, "VWAP": df_5m['VWAP3'], "uptrends": df_uptrends,
                  "downtrends": df_downtrends, "firstBars": df_5m[df_5m.index >= day_data.day_open][0:6],
                  "ts_oii": ts_oii, "ts_is_high_lh": ts_is_high_lh, "ts_is_high_oc": ts_is_high_oc,
                  "df_bracket_moves": df_bracket_moves, "df_bracket_moves_after_next": df_bracket_moves_after_next}
      with open(f'{directory_evals}/{symbol}_{date_str}.pkl', "wb") as f:
        pickle.dump(metadata, f)

      print(f'{symbol} finished {date_str}')
      #%%
      prior_day = day_start
      day_start = day_start + timedelta(days=1)
