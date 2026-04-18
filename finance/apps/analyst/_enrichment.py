"""
finance.apps.analyst._enrichment
==================================
Enrich scanner candidates with technical indicators from IBKR daily data.

Computes SMAs, slopes, Bollinger Bands, ATR, RS vs SPY, RVOL, and VDU.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from finance.apps.analyst._models import Candidate, EnrichedCandidate, TechnicalData
from finance.apps.conditions._data import classify_slope, load_daily

log = logging.getLogger(__name__)

SLOPE_LOOKBACK = 10
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
RVOL_PERIOD = 50
VDU_LOOKBACK = 10
VDU_THRESHOLD = 0.7  # volume < 70% of average = contracting
RS_SLOPE_LOOKBACK = 10
TRADING_DAYS_PER_YEAR = 252


def enrich(candidates: list[Candidate]) -> list[EnrichedCandidate]:
    """Enrich all candidates with technical data. Loads SPY once for RS computation."""
    spy_df = load_daily("SPY")
    if spy_df is None:
        log.warning("SPY data not available — RS computation will be skipped")

    results: list[EnrichedCandidate] = []
    for c in candidates:
        df = load_daily(c.symbol)
        if df is None or len(df) < 50:
            log.debug("No/insufficient data for %s", c.symbol)
            results.append(EnrichedCandidate(candidate=c, data_available=False))
            continue

        technicals = _compute_technicals(c.symbol, df, spy_df)
        results.append(EnrichedCandidate(
            candidate=c, technicals=technicals, data_available=True,
        ))

    available = sum(1 for e in results if e.data_available)
    log.info("Enriched %d/%d candidates with IBKR data", available, len(candidates))
    return results


def _compute_technicals(
    symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame | None,
) -> TechnicalData:
    """Compute all technical indicators for a single symbol."""
    close = df["c"]

    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(BB_PERIOD).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    last_price = close.iloc[-1]

    # Bollinger Band width
    bb_std = close.rolling(BB_PERIOD).std()
    bb_upper = sma_20 + BB_STD * bb_std
    bb_lower = sma_20 - BB_STD * bb_std
    bb_width = (bb_upper - bb_lower) / sma_20
    bb_width_current = bb_width.iloc[-1] if not bb_width.empty else None
    bb_width_avg = bb_width.rolling(BB_PERIOD).mean().iloc[-1] if len(bb_width) >= BB_PERIOD else None

    # ATR(14)
    atr = _compute_atr(df, ATR_PERIOD)

    # 52W high/low
    year_slice = close.iloc[-TRADING_DAYS_PER_YEAR:] if len(close) >= TRADING_DAYS_PER_YEAR else close
    high_52w = year_slice.max()
    low_52w = year_slice.min()

    # 12-month return
    return_12m = None
    if len(close) >= TRADING_DAYS_PER_YEAR:
        return_12m = (last_price / close.iloc[-TRADING_DAYS_PER_YEAR] - 1) * 100

    # RS vs SPY
    rs_ratio, rs_slope = _compute_rs(close, spy_df)

    # RVOL
    vol = df["v"] if "v" in df.columns else None
    rvol = None
    volume_contracting = None
    if vol is not None and len(vol) >= RVOL_PERIOD:
        avg_vol = vol.rolling(RVOL_PERIOD).mean()
        rvol = vol.iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else None

        # VDU: recent volume contracting vs average
        recent_avg = vol.iloc[-VDU_LOOKBACK:].mean()
        longer_avg = avg_vol.iloc[-1]
        volume_contracting = (recent_avg / longer_avg) < VDU_THRESHOLD if longer_avg > 0 else None

    return TechnicalData(
        sma_5=_safe_last(sma_5),
        sma_10=_safe_last(sma_10),
        sma_20=_safe_last(sma_20),
        sma_50=_safe_last(sma_50),
        sma_200=_safe_last(sma_200),
        sma_50_slope=classify_slope(sma_50) if sma_50.notna().sum() >= SLOPE_LOOKBACK else None,
        sma_200_slope=classify_slope(sma_200) if sma_200.notna().sum() >= SLOPE_LOOKBACK else None,
        bb_width=bb_width_current,
        bb_width_avg_20=bb_width_avg,
        atr_14=atr,
        high_52w=high_52w,
        low_52w=low_52w,
        return_12m=return_12m,
        rs_vs_spy=rs_ratio,
        rs_slope_10d=rs_slope,
        rvol=rvol,
        volume_contracting=volume_contracting,
    )


def _compute_atr(df: pd.DataFrame, period: int) -> float | None:
    """Compute Average True Range over the given period."""
    if len(df) < period + 1:
        return None
    high = df["h"]
    low = df["l"]
    prev_close = df["c"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return _safe_last(atr)


def _compute_rs(
    close: pd.Series, spy_df: pd.DataFrame | None,
) -> tuple[float | None, float | None]:
    """Compute RS ratio vs SPY and its 10-day slope."""
    if spy_df is None or len(spy_df) < RS_SLOPE_LOOKBACK:
        return None, None

    spy_close = spy_df["c"]

    # Align dates
    aligned = pd.DataFrame({"stock": close, "spy": spy_close}).dropna()
    if len(aligned) < RS_SLOPE_LOOKBACK:
        return None, None

    rs_line = aligned["stock"] / aligned["spy"]
    rs_ratio = rs_line.iloc[-1]

    # 10-day slope as percentage change
    if len(rs_line) >= RS_SLOPE_LOOKBACK:
        rs_recent = rs_line.iloc[-RS_SLOPE_LOOKBACK:]
        rs_slope = (rs_recent.iloc[-1] / rs_recent.iloc[0] - 1) * 100
    else:
        rs_slope = None

    return rs_ratio, rs_slope


def _safe_last(series: pd.Series) -> float | None:
    """Return the last non-NaN value or None."""
    if series.empty:
        return None
    val = series.iloc[-1]
    return float(val) if not np.isnan(val) else None
