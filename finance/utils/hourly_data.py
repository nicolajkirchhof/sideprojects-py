import string
from datetime import datetime

import dateutil
import pandas as pd

from finance import utils


class HourlyData:
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
    self.df_1h = pd.DataFrame()
    self._process_dataframes()

  def _process_dataframes(self) -> None:
    """Process and filter DataFrames for different time periods"""
    # Filter data
    cache = utils.influx.get_candles_range_raw(dateutil.parser.parse('2000-01-01').replace(tzinfo=self.exchange.tz), datetime.now(self.exchange.tz), self.symbol, True)
    self.df_1h = utils.indicators.basic_indicators_from_df(cache)
