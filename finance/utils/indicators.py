import pandas as pd
import numpy as np
from finance import utils


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

def trading_day_moves(day_data, df_candles, pullback_threshold_multiplier = 0.3):

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

  start = df_candles.iloc[0]
  low = df_candles['l'].min()
  extreme_low = {"ts": df_candles['l'].idxmin(), "type": 'l', "value": low, **calculate_offssets(low)}
  high = df_candles['h'].max()
  extreme_high = {"ts": df_candles['h'].idxmax(), "type": 'h', "value": high, **calculate_offssets(high)}
  end = df_candles.iloc[-1]
  is_up_move_day = extreme_low['ts'] < extreme_high['ts']
  extrema = [extreme_low, extreme_high] if is_up_move_day else [extreme_high, extreme_low]
  is_open_extreme = extrema[0]['ts'] == start.name
  is_close_extreme = extrema[-1]['ts'] == end.name
  if not is_open_extreme:
    extrema.insert(0, {"ts": df_candles.index[0], "type": 'o',
                       "value": df_candles.o.iloc[0], **calculate_offssets(df_candles.o.iloc[0])})
  if not is_close_extreme:
    extrema.append({"ts": df_candles.index[-1], "type": 'c',
                    "value": df_candles.c.iloc[-1], **calculate_offssets(df_candles.c.iloc[-1])})
  df_extrema = pd.DataFrame(extrema)
  ##%%
  vwap_tops_filter = ((df_candles['VWAP3'].shift(1) < df_candles['VWAP3']) &
                      (df_candles['VWAP3'] > df_candles['VWAP3'].shift(-1)))
  vwap_bottoms_filter = ((df_candles['VWAP3'].shift(1) > df_candles['VWAP3']) &
                         (df_candles['VWAP3'] < df_candles['VWAP3'].shift(-1)))
  df_candles['VWAP3_is_top'] = vwap_tops_filter
  df_candles['VWAP3_is_bottom'] = vwap_bottoms_filter
  vwap_tops_index = df_candles[vwap_tops_filter].index.tolist()
  vwap_bottoms_index = df_candles[vwap_bottoms_filter].index.tolist()
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
    extrema_range_filter = ((df_candles.index >= start_extrema['ts']) &
                            (df_candles.index < end_extrema['ts']))
    df_dir = df_candles[extrema_range_filter]
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

def basic_indicators_from_df(df):
  df['VWAP3'] = (df['c'] + df['h'] + df['l']) / 3
  df['20EMA'] = df['VWAP3'].ewm(span=20, adjust=False).mean()
  df['200EMA'] = df['VWAP3'].ewm(span=200, adjust=False).mean()
  df['lh'] = (df.h - df.l)
  df['oc'] = (df.c - df.o)
  df['chg'] = df.c - df.o
  df['pc'] = utils.pct.percentage_change_array(df.o, df.c)
  df.dropna(subset=['o', 'h', 'c', 'l'], inplace=True)
  return df
def hurst(ts):
  lags = range(2, 20)
  tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
  poly = np.polyfit(np.log(lags), np.log(tau), 1)
  return poly[0] * 2.0

def rolling_autocorr(series, window, lag):
  """
  Calculates the rolling autocorrelation of log returns.
  """
  # Calculate Log Returns (Literature Standard)
  log_returns = np.log(series / series.shift(1))

  # Calculate rolling correlation between returns and their lagged version
  return log_returns.rolling(window=window).corr(log_returns.shift(lag))

def get_composite_autocorr(series, window=20, max_lag=3):
  """
  Calculates the average autocorrelation across multiple lags
  to filter out single-day noise.
  """
  log_returns = np.log(series / series.shift(1))
  corrs = []
  for l in range(1, max_lag + 1):
    corrs.append(log_returns.rolling(window=window).corr(log_returns.shift(l)))

  # Return the average of Lags 1, 2, and 3
  return pd.concat(corrs, axis=1).mean(axis=1)

def swing_indicators(df_stk, lrc = [50, 100, 200], timeframe='D'):
  df_stk.sort_index(inplace=True)
  df_stk['gap'] = df_stk.o - df_stk.shift().c
  df_stk['gappct'] = utils.pct.percentage_change_array(df_stk.shift().c, df_stk.o).fillna(0)
  df_stk['pct'] = utils.pct.percentage_change_array(df_stk.shift().c, df_stk.c).fillna(0)
  df_stk['chg'] = (df_stk.c - df_stk.shift().c).fillna(0)
  df_stk['vwap3'] = (df_stk['c']+df_stk['h']+df_stk['l'])/3
  if 'iv' not in df_stk.columns:
    df_stk['iv'] = np.nan

  # RVOL Calculation (Relative Volume)
  # 20-day is standard for swing trading
  for vol in [9, 20, 50]:
    df_stk[f'v{vol}'] = df_stk['v'].rolling(window=vol).mean()
    df_stk[f'rvol{vol}'] = df_stk['v'] / df_stk[f'v{vol}']

  # 2. Multi-Lag Autocorrelations (Legs)
  # Lag 1: Daily momentum/mean-reversion
  # Lag 5: Weekly cycle persistence
  # Lag 21: Monthly cycle (Institutional window)
  for window in [100]:
    for lag in [1, 5, 10, 20]:
      df_stk[f'ac{window}_lag_{lag}'] = rolling_autocorr(df_stk['c'], window=window, lag=lag)

  # --- Implementation on a DataFrame 'df' ---
  # 1. Standard Momentum (20-day, Lag-1)
  # Use this to confirm trend persistence.
  df_stk['ac_mom'] = rolling_autocorr(df_stk['c'], window=20, lag=1)

  # 2. Short-Term Mean Reversion (10-day, Lag-1)
  # Look for values < -0.2 for "snap-back" opportunities.
  df_stk['ac_mr'] = rolling_autocorr(df_stk['c'], window=10, lag=1)

  # # 3. Institutional "Hidden" Momentum (60-day, Lag-5)
  # # If this is higher than Lag-1, institutions are likely accumulating.
  df_stk['ac_inst'] = rolling_autocorr(df_stk['c'], window=50, lag=5)

  # 4. Composite Swing Signal (20-day, Avg Lags 1-3)
  # Best for general trend-following filters.
  df_stk['ac_comp'] = get_composite_autocorr(df_stk['c'], window=20, max_lag=3)

  # Historic Volatility (Annualized)
  # Standard formula: std(log_returns) * sqrt(252)
  log_ret = np.log(df_stk['c'] / df_stk['c'].shift(1))
  for days in [9, 14, 20, 50]:
    df_stk[f'hv{days}'] = log_ret.rolling(window=days).std(ddof=0) * np.sqrt(252)
  df_stk['std'] = df_stk['c'].rolling(window=20).std(ddof=0)
  df_stk['std_mv'] = df_stk['chg'].abs() / df_stk['std']

    # ATR Calculation
  df_stk['hl'] = df_stk['h'] - df_stk['l']
  df_stk['hpc'] = np.abs(df_stk['h'] - df_stk['c'].shift())
  df_stk['lpc'] = np.abs(df_stk['l'] - df_stk['c'].shift())

  tr = df_stk[['hl', 'hpc', 'lpc']].max(axis=1)
  for atr in [1, 9, 14, 20, 50]:
    df_stk[f'atr{atr}'] = tr.ewm(alpha=1/atr, adjust=False).mean()
    df_stk[f'atrp{atr}'] = (df_stk[f'atr{atr}'] / df_stk['c']) * 100

  for ema in [5, 10, 20, 50, 100, 200]:
    df_stk[f'ema{ema}'] = df_stk['vwap3'].ewm(span=ema, adjust=False).mean()
    df_stk[f'ema{ema}_slope'] = df_stk[f'ema{ema}'].diff()
    df_stk[f'ema{ema}_dist'] = ((df_stk['c'] - df_stk[f'ema{ema}']) / df_stk[f'ema{ema}']) * 100
    df_stk[f'ema{ema}_dist_atr'] = (df_stk['c'] - df_stk[f'ema{ema}']) / df_stk[f'atr{atr}']

    # Hurst Exponent (Trend vs Mean Reversion)
  # Calculated over a rolling 100-day window
  # H < 0.5: Mean Reverting | H > 0.5: Trending
  df_stk['hurst50'] = df_stk['c'].rolling(window=50).apply(hurst, raw=True)
  df_stk['hurst100'] = df_stk['c'].rolling(window=100).apply(hurst, raw=True)

  df_stk['year'] = df_stk.index.year
  df_stk['month'] = df_stk.index.month
  df_stk['day'] = df_stk.index.day

  # Consecutive Up/Down Days
  # 1 if up, -1 if down, 0 if flat
  change_sign = np.sign(df_stk['pct'].fillna(0))
  # Create groups: a new group starts every time the sign changes
  streak_groups = (change_sign != change_sign.shift()).cumsum()
  # Cumulate the signs within each group
  df_stk['streak'] = change_sign.groupby(streak_groups).cumsum()

  # Keltner Channels (20 EMA + 1.5 * ATR)
  df_stk['kc_basis'] = df_stk['c'].ewm(span=20, adjust=False).mean()
  df_stk['kc_upper'] = df_stk['kc_basis'] + (df_stk['atr20'] * 1.5)
  df_stk['kc_lower'] = df_stk['kc_basis'] - (df_stk['atr20'] * 1.5)

  # Donchian Channel Midline (20-day High/Low Midpoint)
  df_stk['dc_upper'] = df_stk['h'].rolling(window=20).max()
  df_stk['dc_lower'] = df_stk['l'].rolling(window=20).min()
  df_stk['dc_mid'] = (df_stk['dc_upper'] + df_stk['dc_lower']) / 2

  # Bollinger Bands (for TTM Squeeze)
  df_stk['bb_basis'] = df_stk['c'].rolling(window=20).mean()
  df_stk['bb_std'] = df_stk['c'].rolling(window=20).std(ddof=0)
  df_stk['bb_upper'] = df_stk['bb_basis'] + (df_stk['bb_std'] * 2)
  df_stk['bb_lower'] = df_stk['bb_basis'] - (df_stk['bb_std'] * 2)

  # TTM Squeeze
  # Squeeze is ON when Bollinger Bands are inside Keltner Channels
  df_stk['squeeze_on'] = (df_stk['bb_lower'] > df_stk['kc_lower']) & (df_stk['bb_upper'] < df_stk['kc_upper'])

  # TTM Momentum Histogram
  def ttm_mom(subset):
    # Linear regression of Price vs Average of (High/Low midline + SMA)
    y = subset
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return slope * (len(y) - 1) + intercept

  # Use a 20-period lookback for the midline
  midline_avg = (df_stk['dc_mid'] + df_stk['bb_basis']) / 2
  mom_val = df_stk['c'] - midline_avg
  df_stk['ttm_mom'] = mom_val.rolling(window=20).apply(ttm_mom, raw=True)

  return df_stk


def linear_regression_channel(df, window):
  def lin_reg(subset):
    y = subset
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return slope * (len(y) - 1) + intercept

  base_col = f'lrc{window}'
  df[base_col] = df['c'].rolling(window=window).apply(lin_reg, raw=True)
  rolling_std = df['c'].rolling(window=window).std(ddof=0)
  df[f'{base_col}_ub'] = df[base_col] + (rolling_std * 2)
  df[f'{base_col}_lb'] = df[base_col] - (rolling_std * 2)

  # Distance from LRC lines (Percentage)
  df[f'{base_col}_dist'] = ((df['c'] - df[base_col]) / df[base_col]) * 100
  df[f'{base_col}_ub_dist'] = ((df['c'] - df[f'{base_col}_ub']) / df[f'{base_col}_ub']) * 100
  df[f'{base_col}_lb_dist'] = ((df['c'] - df[f'{base_col}_lb']) / df[f'{base_col}_lb']) * 100
  return df
