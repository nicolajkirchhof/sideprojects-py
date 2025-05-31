import string
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from finance import utils


class TradingDayData:
  def __init__(self, symbol: string, min_future_data = timedelta(days=7), min_future_cache = None):
    """
    Initialize trading day data with start time and exchange settings

    Parameters:
    -----------
    day_start : datetime
        The starting day timestamp
    symbol: string
        The sybol to analyze
    """
    self.min_future_data = min_future_data
    self.min_future_cache = min_future_cache if min_future_cache is not None else min_future_data * 5
    self.symbol = symbol
    self.exchange = utils.influx.SYMBOLS[symbol]['EX']
    self.df_5m_cache = pd.DataFrame()
    self.df_day_cache = pd.DataFrame()
    self.df_30m_cache = pd.DataFrame()
    self.day_start = None
    self.day_end = None
    self.day_open = None
    self.day_close = None
    self.prior_day = None
    self.last_saturday = None

  def update(self, day_start, prior_day=None) -> None:
    # Calculate time boundaries
    self.day_start = day_start
    self.day_end = day_start + self.exchange['Close'] + timedelta(hours=1)
    self.day_open = day_start + self.exchange['Open']
    self.day_close = day_start + self.exchange['Close']
    self.prior_day = prior_day if prior_day is not None else day_start - timedelta(days=1)
    self.last_saturday = self._get_last_saturday()

    # Store DataFrames
    self._process_dataframes()

    # Calculate candle data
    if self.has_sufficient_data():
      self._calculate_candle_data()

  def _get_last_saturday(self) -> datetime:
    """Get the last Saturday before day_start"""
    days_since_saturday = self.day_start.weekday() - 5  # 5 is Saturday
    if days_since_saturday <= 0:
      days_since_saturday += 7
    return self.day_start - timedelta(days=days_since_saturday)

  def _process_dataframes(self) -> None:
    """Process and filter DataFrames for different time periods"""
    # Filter 5-minute data
    if self.df_day_cache.empty or self.df_day_cache.iloc[-1].name < self.day_start + self.min_future_data:
      self.df_5m_cache, self.df_30m_cache, self.df_day_cache = utils.influx.get_5m_30m_day_date_range_with_indicators(
        self.day_start, self.day_start+self.min_future_cache, self.symbol)
    self.df_5m = self.df_5m_cache[
      (self.df_5m_cache.index >= self.day_start + self.exchange['Open'] - timedelta(hours=1)) &
      (self.df_5m_cache.index <= self.day_end)
      ].copy()

    # Filter 30-minute data
    self.df_30m = self.df_30m_cache[
      (self.df_30m_cache.index >= self.day_start + self.exchange['Open'] - timedelta(hours=1)) &
      (self.df_30m_cache.index <= self.day_end)
      ].copy()

    # Filter daily data
    self.df_day = self.df_day_cache[
      (self.df_day_cache.index >= self.day_start - timedelta(days=21)) &
      (self.df_day_cache.index <= self.day_start + timedelta(days=7))
      ]

    # Adjust day_open and day_close based on available data
    if self.day_open < self.df_5m.index.min():
      self.day_open = self.df_5m.index.min()
    if self.day_close > self.df_5m.index.max():
      self.day_close = self.df_5m.index.max()

    # Calculate period-specific candles
    self.current_week_candle = self.df_5m_cache[
      (self.df_5m_cache.index >= self.last_saturday) &
      (self.df_5m_cache.index <= self.prior_day + self.exchange['Close'])
      ]

    self.prior_week_candle = self.df_5m_cache[
      (self.df_5m_cache.index >= self.last_saturday - timedelta(days=7)) &
      (self.df_5m_cache.index <= self.last_saturday)
      ]

    self.prior_day_candle = self.df_5m_cache[
      (self.df_5m_cache.index >= self.prior_day + self.exchange['Open']) &
      (self.df_5m_cache.index <= self.prior_day + self.exchange['Close'])
      ]

    self.overnight_candle = self.df_5m_cache[
      (self.df_5m_cache.index >= self.day_start) &
      (self.df_5m_cache.index <= self.day_start + self.exchange['Open'] - timedelta(hours=1))
      ]

  def _calculate_candle_data(self) -> None:
    """Calculate various candle data points"""
    intraday_filter = (self.df_5m.index >= self.day_open) & (self.df_5m.index <= self.day_close)

    # Current day data
    self.cdh = self.df_5m[intraday_filter].h.max()
    self.cdl = self.df_5m[intraday_filter].l.min()
    self.cdo = self.df_5m[intraday_filter].o.iat[0]
    self.cdc = self.df_5m[intraday_filter].c.iloc[-1]

    # Prior day data
    self.pdh = self.prior_day_candle.h.max() if not self.prior_day_candle.empty else np.nan
    self.pdl = self.prior_day_candle.l.min() if not self.prior_day_candle.empty else np.nan
    self.pdc = self.prior_day_candle.c.iat[-1] if not self.prior_day_candle.empty else np.nan

    # Weekly data
    self.cwh = self.current_week_candle.h.max() if not self.current_week_candle.empty else np.nan
    self.cwl = self.current_week_candle.l.min() if not self.current_week_candle.empty else np.nan
    self.pwh = self.prior_week_candle.h.max()
    self.pwl = self.prior_week_candle.l.min()

    # Overnight data
    self.onh = self.overnight_candle.h.max() if not self.overnight_candle.empty else np.nan
    self.onl = self.overnight_candle.l.min() if not self.overnight_candle.empty else np.nan

  def has_sufficient_data(self) -> bool:
    """Check if there's sufficient data for analysis"""
    return len(self.df_5m) >= 30

  def get_summary(self) -> dict:
    """Return a summary of key data points"""
    return {
      'day_start': self.day_start,
      'day_open': self.day_open,
      'day_close': self.day_close,
      'current_day_high': self.cdh,
      'current_day_low': self.cdl,
      'prior_day_high': self.pdh,
      'prior_day_low': self.pdl,
      'current_week_high': self.cwh,
      'current_week_low': self.cwl,
      'overnight_high': self.onh,
      'overnight_low': self.onl
    }
