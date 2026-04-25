"""
Pure data loading and indicator computation for the conditions dashboard.

All functions are Qt-free and operate on pandas DataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMA_FAST = 20
SMA_SHORT = 50
SMA_LONG = 200
SLOPE_LOOKBACK = 10
SLOPE_THRESHOLD = 0.0005  # 0.05 %

VIX_ELEVATED = 20
VIX_HIGH = 30
VIX_SPIKE_WINDOW = 5
VIX_SPIKE_THRESHOLD = 0.20  # 20 %


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrendStatus:
    symbol: str
    last_price: float
    sma_20: float
    sma_50: float
    sma_200: float
    price_above_20: bool
    price_above_50: bool
    price_above_200: bool
    sma_20_slope: str   # "rising" | "flat" | "falling"
    sma_50_slope: str   # "rising" | "flat" | "falling"
    sma_200_slope: str


@dataclass(frozen=True)
class VixStatus:
    level: float
    zone: str           # "low" | "elevated" | "high"
    direction: str      # "falling" | "rising" | "spiking"
    is_spiking: bool


# ---------------------------------------------------------------------------
# Slope classification
# ---------------------------------------------------------------------------

def classify_slope(sma_series: pd.Series) -> str:
    """Classify the recent slope of an SMA series.

    Uses the percentage change over the last SLOPE_LOOKBACK bars.
    Returns "rising", "falling", or "flat".
    """
    if len(sma_series) < SLOPE_LOOKBACK:
        return "flat"

    tail = sma_series.iloc[-SLOPE_LOOKBACK:]
    pct_change = (tail.iloc[-1] - tail.iloc[0]) / tail.iloc[0]

    if pct_change > SLOPE_THRESHOLD:
        return "rising"
    if pct_change < -SLOPE_THRESHOLD:
        return "falling"
    return "flat"


# ---------------------------------------------------------------------------
# Trend status (E1-S1)
# ---------------------------------------------------------------------------

def compute_trend_status(symbol: str, df: pd.DataFrame) -> TrendStatus | None:
    """Compute SMA trend status for a symbol from daily close data.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "SPY", "QQQ").
    df : pd.DataFrame
        Daily OHLCV with at least a ``c`` (close) column and DatetimeIndex.

    Returns None if insufficient data (< SMA_LONG bars).
    """
    if df is None or df.empty or len(df) < SMA_LONG:
        return None

    close = df["c"]
    last_price = float(close.iloc[-1])

    sma_20_series = close.rolling(SMA_FAST).mean()
    sma_50_series = close.rolling(SMA_SHORT).mean()
    sma_200_series = close.rolling(SMA_LONG).mean()

    sma_20 = float(sma_20_series.iloc[-1])
    sma_50 = float(sma_50_series.iloc[-1])
    sma_200 = float(sma_200_series.iloc[-1])

    return TrendStatus(
        symbol=symbol,
        last_price=last_price,
        sma_20=sma_20,
        sma_50=sma_50,
        sma_200=sma_200,
        price_above_20=last_price > sma_20,
        price_above_50=last_price > sma_50,
        price_above_200=last_price > sma_200,
        sma_20_slope=classify_slope(sma_20_series.dropna()),
        sma_50_slope=classify_slope(sma_50_series.dropna()),
        sma_200_slope=classify_slope(sma_200_series.dropna()),
    )


# ---------------------------------------------------------------------------
# VIX status (E1-S2)
# ---------------------------------------------------------------------------

def compute_vix_status(df: pd.DataFrame) -> VixStatus | None:
    """Compute VIX level, zone, direction, and spike detection.

    Parameters
    ----------
    df : pd.DataFrame
        Daily VIX data with a ``c`` (close) column and DatetimeIndex.

    Returns None if data is empty.
    """
    if df is None or df.empty:
        return None

    close = df["c"]
    level = float(close.iloc[-1])

    # Zone classification
    if level >= VIX_HIGH:
        zone = "high"
    elif level >= VIX_ELEVATED:
        zone = "elevated"
    else:
        zone = "low"

    # Spike detection: >20% increase over 5 trading days
    is_spiking = False
    if len(close) > VIX_SPIKE_WINDOW:
        ref = float(close.iloc[-(VIX_SPIKE_WINDOW + 1)])
        if ref > 0:
            is_spiking = (level / ref - 1) > VIX_SPIKE_THRESHOLD

    # Direction
    if is_spiking:
        direction = "spiking"
    elif len(close) >= 2 and float(close.iloc[-1]) < float(close.iloc[-2]):
        direction = "falling"
    else:
        direction = "rising"

    return VixStatus(
        level=level,
        zone=zone,
        direction=direction,
        is_spiking=is_spiking,
    )


# ---------------------------------------------------------------------------
# Composite GO / NO-GO
# ---------------------------------------------------------------------------

def compute_go_nogo(
    spy_trend: TrendStatus | None,
    vix: VixStatus | None,
) -> str:
    """Derive composite market regime status.

    Returns "GO", "CAUTION", or "NO-GO".

    NO-GO triggers (any one):
      - SPY below both SMAs
      - VIX zone is "high" (>=30)
      - VIX is spiking (>20% in 5d)
      - Missing SPY trend data

    GO requires ALL of:
      - SPY above both SMAs
      - 200d SMA rising
      - VIX zone is "low" (<20)
      - VIX not spiking

    Everything else is CAUTION.
    """
    if spy_trend is None:
        return "NO-GO"

    # NO-GO checks
    nogo = (
        (not spy_trend.price_above_50 and not spy_trend.price_above_200)
        or (vix is not None and vix.zone == "high")
        or (vix is not None and vix.is_spiking)
    )
    if nogo:
        return "NO-GO"

    # GO checks — all must be true
    go = (
        spy_trend.price_above_50
        and spy_trend.price_above_200
        and spy_trend.sma_200_slope == "rising"
        and vix is not None
        and vix.zone == "low"
        and not vix.is_spiking
    )
    if go:
        return "GO"

    return "CAUTION"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_daily(symbol: str) -> pd.DataFrame | None:
    """Load cached daily OHLCV for a symbol from the IBKR parquet cache.

    Returns None if the cache file doesn't exist.
    """
    from finance.utils.ibkr import daily_w_volatility

    df = daily_w_volatility(symbol, offline=True)
    if df is None or df.empty:
        return None
    return df
