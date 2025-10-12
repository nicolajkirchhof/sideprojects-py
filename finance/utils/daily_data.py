import string

import pandas as pd

from finance import utils


class DailyData:
  def __init__(self, symbol: string):
    """
    Initialize trading day data with start time and exchange settings

    Parameters:
    -----------
    day_start : datetime
        The starting day timestamp
    symbol: string
        The sybol to analyze
    """
    self.symbol = symbol
    self.exchange = utils.influx.SYMBOLS[symbol]['EX']
    self.df_day = pd.DataFrame()
    self._process_dataframes()

  def _process_dataframes(self) -> None:
    """Process and filter DataFrames for different time periods"""
    # Filter data
    cache = utils.influx.get_candles_range_all_aggregate_tz(self.symbol, '1d')
    self.df_day = utils.indicators.basic_indicators_from_df(cache)

    # Weekly resample - weeks start on Sunday
    resample = self.df_day.resample('W-SUN', label='left', closed='left').agg(
      o=('o', 'first'),
      h=('h', 'max'),
      l=('l', 'min'),
      c=('c', 'last')
    ).copy()
    self.df_week = utils.indicators.basic_indicators_from_df(resample)


