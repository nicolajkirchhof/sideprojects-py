import string
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from finance import utils

DATASOURCES = ['dolt', 'barchart', 'ibkr']


class SwingTradingData:
  def __init__(self, symbol: string, datasource='auto', offline=False):
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
    self.datasource = datasource

    match datasource:
      case 'auto':
        self.datasource = 'ibkr'
        self.df_day = utils.ibkr.daily_w_volatility(symbol, offline=offline)
        if self.df_day is None or self.df_day.empty:
          self.df_day = utils.dolt_data.daily_w_volatility(symbol)
          self.datasource = 'dolt'
      case 'dolt':
        self.df_day = utils.dolt_data.daily_w_volatility(symbol)
      case 'barchart':
        self.df_day = utils.barchart_data.daily_w_volatility(symbol)
      case 'ibkr':
        self.df_day = utils.ibkr.daily_w_volatility(symbol, offline=offline)
      case _:
        raise ValueError(f'Invalid datasource: {datasource}')

    if self.df_day is None or self.df_day.empty:
      self.empty = True
      return
    self.empty = False

    self.df_week = self.df_day[utils.definitions.OHLCLV].resample('1W').agg(o=('o', 'first'), h=('h', 'max'),
                                                                            l=('l', 'min'), c=('c', 'last'),
                                                                            v=('v', 'sum')).copy()
    self.df_month = self.df_day[utils.definitions.OHLCLV].resample('1ME').agg(o=('o', 'first'), h=('h', 'max'),
                                                                             l=('l', 'min'), c=('c', 'last'),
                                                                             v=('v', 'sum')).copy()

    self.df_day = utils.indicators.swing_indicators(self.df_day)
    self.df_week = utils.indicators.swing_indicators(self.df_week, [50, 100], timeframe=None)
    self.df_month = utils.indicators.swing_indicators(self.df_month, [50], timeframe=None)
    self.info = utils.dolt_data.symbol_info(symbol)
    self.market_cap = None
    self.df_shares_outstanding = None


    self.df_shares_outstanding = utils.dolt_data.financial_info(symbol)
    if not self.df_shares_outstanding.empty:
      df_market_cap = self.df_shares_outstanding.copy()
      df_market_cap = df_market_cap[~df_market_cap.index.duplicated(keep='first')]
      flt_market_cap = df_market_cap.index >= self.df_day.index.min()      # Align price to shares outstanding index using forward-fill (captures the last 'closed' price)
      if not any(flt_market_cap):
        df_market_cap['c'] = self.df_day['c'].iloc[0]
      else:
        df_market_cap['c'] = self.df_day['c'].reindex(df_market_cap.index[flt_market_cap], method='ffill')
      # Remove any dates that occurred before our available price data
      df_market_cap.dropna(subset=['c'], inplace=True)

      df_market_cap['market_cap'] = df_market_cap['c'] * df_market_cap['shares_outstanding']
      self.market_cap = df_market_cap


