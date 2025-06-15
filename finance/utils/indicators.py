import pandas as pd
import numpy as np


def adaptive_moving_average(prices, period=10, fast=3, slow=30):
  """
    Calculate the Adaptive Moving Average (AMA or KAMA).

    Args:
        prices (pd.Series): The price series (e.g., closing prices).
        period (int): Lookback period for the Efficiency Ratio (ER).
        fast (int): Period for the fast EMA smoothing.
        slow (int): Period for the slow EMA smoothing.

    Returns:
        pd.Series: Adaptive moving average for the given prices.
    """
  # Calculate Fast and Slow smoothing constants
  fast_sc = 2 / (fast + 1)  # Fast EMA smoothing constant
  slow_sc = 2 / (slow + 1)  # Slow EMA smoothing constant

  # Calculate the Efficiency Ratio (ER)
  price_diff = prices.diff(period).abs()  # Absolute price difference
  volatility = prices.diff().abs().rolling(window=period).sum()
  er = price_diff / volatility
  er = er.fillna(0)  # Handle any NaN values, especially for the first `period`

  # Calculate the Smoothing Constant (SC)
  sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

  # Adaptive Moving Average (AMA)
  ama = [prices.iloc[0]]  # Start AMA with the first price
  for i in range(1, len(prices)):
    ama.append(ama[-1] + sc.iloc[i] * (prices.iloc[i] - ama[-1]))

  return pd.Series(ama, index=prices.index)

def trading_day_moves(day_data, use_all_day = True, pullback_threshold_multiplier = 0.3):
  df_5m = day_data.df_5m_ad if use_all_day else day_data.df_5m

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
  extreme_high = {"ts": df_5m['h'].idxmax(), "type": 'h', "value": high, **calculate_offssets(high)}
  end = df_5m.iloc[-1]
  is_up_move_day = extreme_low['ts'] < extreme_high['ts']
  extrema = [extreme_low, extreme_high] if is_up_move_day else [extreme_high, extreme_low]
  is_open_extreme = extrema[0]['ts'] == start.name
  is_close_extreme = extrema[-1]['ts'] == end.name
  if not is_open_extreme:
    extrema.insert(0, {"ts": df_5m.index[0], "type": 'o',
                       "value": df_5m.o.iloc[0], **calculate_offssets(df_5m.o.iloc[0])})
  if not is_close_extreme:
    extrema.append({"ts": df_5m.index[-1], "type": 'c',
                    "value": df_5m.c.iloc[-1], **calculate_offssets(df_5m.c.iloc[-1])})
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
  pullback_threshold = pullback_threshold_multiplier * (day_data.cdh - day_data.cdl)
  key = f'dev{pullback_threshold_multiplier * 100:n}'
  # print(key)
  for i in range(1, len(df_extrema)):
    ##%%
    start_extrema = df_extrema.iloc[i - 1]
    end_extrema = df_extrema.iloc[i]
    direction = 1 if end_extrema.type == 'h' or start_extrema.type == 'l' else -1
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
      is_candidate = (((df_dir.iloc[j]['VWAP3_is_top'] and direction < 0) or (
          df_dir.iloc[j]['VWAP3_is_bottom'] and direction > 0)) and
                      abs(df_dir.iloc[j].VWAP3 - last_pivot.VWAP3) > pullback_threshold)
      # print('Is candidate:', is_candidate)
      if is_candidate:
        is_new = False
        if extrema_candidate is None:
          # Create pullback if last pivot distance is more than PULLBACK_THRESHOLD day range
          extrema_candidate = {"start": last_pivot.name, "o": last_pivot.VWAP3, "type": key, "direction": direction}
          is_new = True

        if is_new or (direction < 0 and extrema_candidate['c'] < df_dir.iloc[j].VWAP3) or (
            direction > 0 and extrema_candidate['c'] > df_dir.iloc[j].VWAP3):
          # Pullback is actually larger, update the range
          # print('Is higher or new candidate')
          extrema_candidate['end'] = df_dir.iloc[j].name
          extrema_candidate['c'] = df_dir.iloc[j].VWAP3

      # Close pivot on lower pivot bottom / higher pivot top or when the next low / high has a distance of greater than the pullback threshold
      is_extrema_candidate_with_pullback_dist = (
          extrema_candidate is not None and abs(extrema_candidate['c'] - df_dir.iloc[j].VWAP3) > pullback_threshold)

      # print('Is pullback dist:', is_extrema_candidate_with_pullback_dist)
      is_pivot_move = (df_dir.iloc[j]['VWAP3_is_bottom'] and direction < 0 and (
          df_dir.iloc[j]['VWAP3'] < last_pivot['VWAP3'] or is_extrema_candidate_with_pullback_dist) or
                       df_dir.iloc[j]['VWAP3_is_top'] and direction > 0 and (df_dir.iloc[j]['VWAP3'] > last_pivot[
            'VWAP3'] or is_extrema_candidate_with_pullback_dist))
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
    extrema_start = {"ts": extrema['start'], "type": extrema['type'], 'value': extrema['o'],
                     **calculate_offssets(extrema['o'])}
    extrema_end = {"ts": extrema['end'], "type": extrema['type'], 'value': extrema['c'],
                   **calculate_offssets(extrema['c'])}
    df_extrema = pd.concat([df_extrema, pd.DataFrame([extrema_start, extrema_end])]).sort_values(by='ts')
    df_extrema.reset_index(drop=True, inplace=True)

  return df_extrema, pullback_threshold

def trends(df_5m, df_extrema):
  # global pullbacks, df_uptrends, df_downtrends
  ##%%
  # Follow algorithm
  # 1. Start with a candle range defined as [start end] whereas start['l'] is the lowest value and end['h'] is the highest value
  # 2. Count every succession of negative candles
  ##%%
  # pullback = {"start": , "end": , "duration": , "h": , "l": , "o": , "c": , "type": 'SC', "direction": 1 | -1"}"
  ##%%
  pullbacks = []
  for i in range(1, len(df_extrema[df_extrema.type.isin(['o', 'h', 'c', 'l', 'dev2'])])):
    start_extrema = df_extrema.iloc[i - 1]
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
        current_pullback['end'] = df_dir.iloc[j - 1].name
        current_pullback['l'] = df_dir.iloc[j_start:j].l.min()
        current_pullback['h'] = df_dir.iloc[j_start:j].h.max()
        current_pullback['c'] = df_dir.iloc[j - 1]['c']

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
        current_downtrend = create_trend(i, 'h')
      elif df_5m.iloc[i].h < current_downtrend['c']:
        current_downtrend['end'] = df_5m.iloc[i].name
        current_downtrend['c'] = df_5m.iloc[i].h
        current_downtrend['ema20_c'] = df_5m.iloc[i]["20EMA"]
        current_downtrend['bars'] += 1
      else:
        if current_downtrend['end'] != current_downtrend['start']:
          downtrends.append(current_downtrend)
        current_downtrend = create_trend(i, 'h')

    df_uptrends = pd.DataFrame(uptrends)
    df_uptrends['trend_support'] = df_uptrends['ema20_o'] < df_uptrends['ema20_c']
    df_downtrends = pd.DataFrame(downtrends)
    if not df_downtrends.empty:
      df_downtrends['trend_support'] = df_downtrends['ema20_o'] > df_downtrends['ema20_c']

    return pullbacks, df_uptrends, df_downtrends
