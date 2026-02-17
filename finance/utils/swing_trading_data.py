import string
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from finance import utils

DATASOURCES = ['dolt', 'barchart', 'ibkr']


class SwingTradingData:
  def __init__(self, symbol: string, datasource='auto', offline=False, metainfo=True):
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
        if self.df_day is None or self.df_day.empty or 'c' not in self.df_day.columns:
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

    if self.df_day is None or self.df_day.empty or 'c' not in self.df_day.columns:
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

    if not metainfo:
      return

    # Calculate Original Price (Unadjusted for Splits)
    self.df_day['original_price'] = self.df_day['c']
    df_splits = utils.dolt_data.splits(symbol)
    if not df_splits.empty:
      df_splits = df_splits.sort_index()
      # Initialize cumulative factor to 1.0
      cum_factor = pd.Series(1.0, index=self.df_day.index)

      # Iterate over splits
      for split_date, row in df_splits.iterrows():
        col_for = 'for_factor' if 'for_factor' in row else 'from_factor'
        denom = float(row[col_for])
        if denom == 0: continue
        ratio = float(row['to_factor']) / denom

        # Apply adjustment to all dates strictly before the split
        mask = cum_factor.index < split_date
        cum_factor.loc[mask] *= ratio

      self.df_day['original_price'] = self.df_day['c'] * cum_factor

    self.info = utils.dolt_data.symbol_info(symbol)
    self.market_cap = None
    self.df_shares_outstanding = None
    self.df_shares_outstanding = utils.dolt_data.financial_info(symbol)
    if not self.df_shares_outstanding.empty:
      df_market_cap = self.df_shares_outstanding.copy()
      df_market_cap = df_market_cap[~df_market_cap.index.duplicated(keep='first')]
      flt_market_cap = df_market_cap.index >= self.df_day.index.min()      # Align price to shares outstanding index using forward-fill (captures the last 'closed' price)
      
      # Use original_price for correct Market Cap calculation
      price_col = 'original_price' if 'original_price' in self.df_day.columns else 'c'

      if not any(flt_market_cap):
        df_market_cap['c'] = self.df_day[price_col].iloc[0]
      else:
        df_market_cap['c'] = self.df_day[price_col].reindex(df_market_cap.index[flt_market_cap], method='ffill')
      # Remove any dates that occurred before our available price data
      df_market_cap.dropna(subset=['c'], inplace=True)

      df_market_cap['market_cap'] = df_market_cap['c'] * df_market_cap['shares_outstanding']

      if df_market_cap.empty: return
      
      # Classify Market Cap using fundamentals utility
      df_market_cap['market_cap_class'] = df_market_cap.apply(
          lambda row: utils.fundamentals.classify_market_cap(row['market_cap'], row.name.year), 
          axis=1
      )
      
      self.market_cap = df_market_cap
