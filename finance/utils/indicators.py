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

