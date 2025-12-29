import string
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from finance import utils

DATASOURCES = ['dolt', 'barchart']


class SwingTradingData:
  def __init__(self, symbol: string, datasource='dolt'):
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
    self.df_day = pd.DataFrame()
    match datasource:
      case 'dolt':
        self.df_day = utils.dolt_data.daily_w_volatility(symbol)
      case 'barchart':
        self.df_day = utils.barchart_data.daily_w_volatility(symbol)
      case _:
        raise ValueError(f'Invalid datasource: {datasource}')

    self.df_week = self.df_day[utils.definitions.OHLCLV].resample('1W').agg(o=('o', 'first'), h=('h', 'max'),
                                                                            l=('l', 'min'), c=('c', 'last'),
                                                                            v=('v', 'sum')).copy()
    self.df_month = self.df_day[utils.definitions.OHLCLV].resample('1ME').agg(o=('o', 'first'), h=('h', 'max'),
                                                                             l=('l', 'min'), c=('c', 'last'),
                                                                             v=('v', 'sum')).copy()

    self.df_day = utils.indicators.swing_indicators(self.df_day)
    self.df_week = utils.indicators.swing_indicators(self.df_week, [50, 100])
    self.df_month = utils.indicators.swing_indicators(self.df_month, [50])
    self.symbol_info = utils.dolt_data.symbol_info(symbol)
    self.market_cap = None
    df_shares_outstanding = utils.dolt_data.shares_outstanding_info(symbol)
    if not df_shares_outstanding.empty:
      df_market_cap = pd.merge(df_shares_outstanding, self.df_day.c, left_index=True, right_index=True)
      df_market_cap['market_cap'] = df_market_cap['c'] * df_market_cap['sharesOutstanding']
      self.market_cap = df_market_cap


