"""
Pure data loading and indicator computation for the Trading Assistant.

All functions are Qt-free and operate on pandas DataFrames.
Migrated from finance.apps.conditions._data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# DRIFT regime constants
# ---------------------------------------------------------------------------

# (name, drawdown_min_pct, drawdown_max_pct, vix_min, vix_max, bp_pct, structure, dte_range)
# Source: InvestingPlaybook.md § DRIFT drawdown scaling framework
_DRIFT_TIERS: list[tuple] = [
    ("Normal",          0,   5,   0,  20, 30, "XYZ 111, short puts",              "45–60"),
    ("Elevated",        5,  10,  20,  30, 40, "XYZ 111, short puts",              "45–90"),
    ("Correction",     10,  20,  25,  40, 55, "XYZ 221, short puts, synthetics",  "60–120"),
    ("Deep Correction", 20, 30,  35,  55, 70, "XYZ 221, synthetics, LEAPS puts",  "90–180"),
    ("Bear",           30, 100,  50, 999, 80, "Spreads only, wide strikes, LEAPS","180–360"),
]

# (symbol, block, registry_tier)
_DRIFT_REGISTRY: list[tuple[str, str, str]] = [
    ("SPY", "Directional", "Core"),
    ("QQQ", "Directional", "Core"),
    ("IWM", "Directional", "Core"),
    ("GLD", "Neutral",     "Core"),
    ("SLV", "Neutral",     "Core"),
]

_DRIFT_IVP_MIN_BARS: int = 20   # minimum IV observations needed to compute IVP
_DRIFT_IVP_WINDOW: int  = 252   # look-back window for IV percentile calculation
_DRIFT_IVP_THRESHOLD: float = 50.0  # minimum IVP to consider selling premium

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
# Trend status
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
# VIX status
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
    """Load daily OHLCV for a symbol, refreshing via IBKR if data is stale.

    Tries to fetch the latest bar when the cache is more than one trading day
    old (refresh_offset_days=1).  Falls back to the cached parquet silently if
    IBKR is not reachable, so the panel always shows the best available data.

    Returns None if no data exists at all.
    """
    import logging
    from finance.utils.ibkr import daily_w_volatility

    log = logging.getLogger(__name__)
    try:
        df = daily_w_volatility(symbol, offline=False, refresh_offset_days=1)
    except Exception:
        log.debug("IBKR unreachable for %s — using cached data", symbol)
        df = daily_w_volatility(symbol, offline=True)
    if df is None or df.empty:
        return None
    return df


# ---------------------------------------------------------------------------
# DRIFT regime dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftTier:
    """Current DRIFT regime tier derived from SPY drawdown × VIX level."""
    name: str               # "Normal" | "Elevated" | "Correction" | "Deep Correction" | "Bear"
    drawdown_pct: float     # current SPY drawdown from 52W high (positive %)
    bp_pct: int             # recommended DRIFT buying-power allocation %
    structure: str          # recommended structure text
    dte_range: str          # DTE range text (e.g. "45–60")


@dataclass(frozen=True)
class DriftUnderlyingStatus:
    """Eligibility and structure recommendation for a single DRIFT underlying."""
    symbol: str
    block: str              # "Directional" | "Neutral"
    registry_tier: str      # "Core" | "Selective" | "Optional"
    ivp: float | None       # IV percentile 0–100, or None if data unavailable
    price_above_200: bool | None
    iv_gt_hv: bool | None   # True when current IV > last-known HV
    eligible: bool          # True when IVP ≥ _DRIFT_IVP_THRESHOLD
    structure: str          # e.g. "Short put / XYZ" | "Spread only" | "Wait — IVP < 50"


# ---------------------------------------------------------------------------
# DRIFT regime computation
# ---------------------------------------------------------------------------


def _tier_index_from_drawdown(drawdown_pct: float) -> int:
    """Return the tier index (0=Normal … 4=Bear) for a given drawdown %."""
    for i, row in enumerate(_DRIFT_TIERS):
        _, dd_min, dd_max, *_ = row
        if drawdown_pct < dd_max:
            return i
    return len(_DRIFT_TIERS) - 1


def _tier_index_from_vix(vix_level: float) -> int:
    """Return the tier index for a given VIX level."""
    for i, row in enumerate(_DRIFT_TIERS):
        _, _dd_min, _dd_max, vix_min, vix_max, *_ = row
        if vix_level < vix_max:
            return i
    return len(_DRIFT_TIERS) - 1


def compute_drift_tier(
    spy_df: pd.DataFrame | None,
    vix: VixStatus | None,
) -> DriftTier:
    """
    Compute the DRIFT regime tier from SPY drawdown and VIX level.

    Both the drawdown and VIX must signal the same or higher tier to advance.
    When they disagree the more conservative (lower-severity) tier is used —
    consistent with InvestingPlaybook.md: "Both conditions must be met to
    advance a tier."

    When data is missing returns the Normal default (safe, deployable state).

    Parameters
    ----------
    spy_df:
        Daily OHLCV + IV/HV DataFrame for SPY. Needs ≥ SMA_LONG bars with
        a ``c`` (close) column.
    vix:
        Pre-computed VixStatus, or None if unavailable.

    Returns
    -------
    DriftTier
        Populated with tier name, drawdown %, BP%, structure, DTE range.
    """
    _normal = DriftTier(
        name="Normal", drawdown_pct=0.0, bp_pct=30,
        structure="XYZ 111, short puts", dte_range="45–60",
    )

    if spy_df is None or spy_df.empty or len(spy_df) < SMA_LONG:
        return _normal

    close = spy_df["c"]
    current = float(close.iloc[-1])
    high_252 = float(close.rolling(_DRIFT_IVP_WINDOW).max().iloc[-1])
    drawdown_pct = max(0.0, (high_252 - current) / high_252 * 100.0)

    vix_level = vix.level if vix is not None else 0.0

    tier_idx = min(
        _tier_index_from_drawdown(drawdown_pct),
        _tier_index_from_vix(vix_level),
    )

    name, _, _, _, _, bp_pct, structure, dte_range = _DRIFT_TIERS[tier_idx]
    return DriftTier(
        name=name,
        drawdown_pct=round(drawdown_pct, 2),
        bp_pct=bp_pct,
        structure=structure,
        dte_range=dte_range,
    )


def compute_drift_underlying(
    symbol: str,
    df: pd.DataFrame | None,
    block: str,
    registry_tier: str,
) -> DriftUnderlyingStatus:
    """
    Compute eligibility and structure recommendation for a single DRIFT underlying.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. "SPY").
    df:
        Daily OHLCV + IV/HV DataFrame from ``load_daily()``.
        Expected columns: ``c`` (close), ``iv`` (implied vol), ``hv`` (hist vol).
    block:
        Registry block — "Directional" or "Neutral".
    registry_tier:
        Registry tier — "Core", "Selective", or "Optional".

    Returns
    -------
    DriftUnderlyingStatus
        IVP and eligibility populated when data is available; all ``None`` and
        ``structure="IBKR required"`` when data is absent.
    """
    _no_data = DriftUnderlyingStatus(
        symbol=symbol, block=block, registry_tier=registry_tier,
        ivp=None, price_above_200=None, iv_gt_hv=None,
        eligible=False, structure="IBKR required",
    )

    if df is None or df.empty or len(df) < SMA_LONG:
        return _no_data

    close = df["c"]
    sma_200 = float(close.rolling(SMA_LONG).mean().iloc[-1])
    current_price = float(close.iloc[-1])
    price_above_200 = current_price > sma_200

    # IV percentile requires the iv column with enough observations
    if "iv" not in df.columns:
        return _no_data

    iv_series = df["iv"].dropna()
    if len(iv_series) < _DRIFT_IVP_MIN_BARS:
        return _no_data

    current_iv = float(iv_series.iloc[-1])
    iv_window = iv_series.iloc[-_DRIFT_IVP_WINDOW:]
    ivp = float((iv_window <= current_iv).mean() * 100.0)

    # IV vs HV — last non-NaN HV value (final bar is often NaN from IBKR)
    iv_gt_hv: bool | None = None
    if "hv" in df.columns:
        hv_valid = df["hv"].dropna()
        if not hv_valid.empty:
            iv_gt_hv = current_iv > float(hv_valid.iloc[-1])

    eligible = ivp >= _DRIFT_IVP_THRESHOLD

    if not eligible:
        structure = f"Wait — IVP {ivp:.0f} < 50"
    elif not price_above_200:
        structure = "Spread only"
    else:
        structure = "Short put / XYZ"

    return DriftUnderlyingStatus(
        symbol=symbol,
        block=block,
        registry_tier=registry_tier,
        ivp=round(ivp, 1),
        price_above_200=price_above_200,
        iv_gt_hv=iv_gt_hv,
        eligible=eligible,
        structure=structure,
    )
