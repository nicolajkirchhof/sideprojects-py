import string
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from finance import utils


class SwingTradingData:
  def __init__(self, symbol: string, datasource = 'dolt'):
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
    self.df_day = utils.dolt_data.daily_w_volatility(symbol)
    self.df_week = self.df_day.resample('1W').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'), v=('v', 'sum')).copy()
    self.df_month = self.df_day.resample('1M').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'), v=('v', 'sum')).copy()

    self.df_day = utils.indicators.swing_indicators(self.df_day)
    self.df_week = utils.indicators.swing_indicators(self.df_week, [50, 100])
    self.df_month = utils.indicators.swing_indicators(self.df_week, [50])

