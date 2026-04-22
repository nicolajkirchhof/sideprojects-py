"""
finance.apps.assistant._market
================================
Market context instrument loading for the daily archive.

Loads EOD data for 13 instruments (indices, volatility, commodities, bonds, forex)
via SwingTradingData and maps swing_indicators output to the unified archive schema.

Error policy: individual instrument failures are logged and skipped.
              A zero or missing price is treated as a load failure and triggers
              a warning (price == 0 is physically impossible for these instruments).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from finance.utils.swing_trading_data import SwingTradingData

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument registry
# ---------------------------------------------------------------------------

#: All market context instruments included in the daily archive.
MARKET_INSTRUMENTS: list[dict[str, str]] = [
    {"symbol": "SPY",    "category": "Indices",             "stk_symbol": "SPY"},
    {"symbol": "QQQ",    "category": "Indices",             "stk_symbol": "QQQ"},
    {"symbol": "IWM",    "category": "Indices",             "stk_symbol": "IWM"},
    {"symbol": "VIX",    "category": "Volatility",          "stk_symbol": "$VIX"},
    {"symbol": "USO",    "category": "Commodities-Energy",  "stk_symbol": "USO"},
    {"symbol": "UNG",    "category": "Commodities-Energy",  "stk_symbol": "UNG"},
    {"symbol": "GLD",    "category": "Commodities-Metals",  "stk_symbol": "GLD"},
    {"symbol": "SLV",    "category": "Commodities-Metals",  "stk_symbol": "SLV"},
    {"symbol": "TLT",    "category": "Bonds",               "stk_symbol": "TLT"},
    {"symbol": "HYG",    "category": "Bonds",               "stk_symbol": "HYG"},
    {"symbol": "EURUSD", "category": "Forex",               "stk_symbol": "^EURUSD"},
    {"symbol": "GBPUSD", "category": "Forex",               "stk_symbol": "^GBPUSD"},
    {"symbol": "USDJPY", "category": "Forex",               "stk_symbol": "^USDJPY"},
]

# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | Any:
    """Return pd.NA if val is None, NaN, or falsy in a numeric sense."""
    if val is None:
        return pd.NA
    try:
        f = float(val)
        if pd.isna(f):
            return pd.NA
        return f
    except (TypeError, ValueError):
        return pd.NA


def _swing_row_to_archive(
    symbol: str,
    category: str,
    last: pd.Series,
    df_day: pd.DataFrame,
) -> dict:
    """
    Map a single df_day.iloc[-1] row (from swing_indicators) to an archive dict.

    Parameters
    ----------
    symbol:
        Archive symbol name (e.g. 'SPY', 'VIX').
    category:
        Archive category (e.g. 'Indices', 'Forex').
    last:
        The last row of df_day (iloc[-1]) after swing_indicators has run.
    df_day:
        Full daily DataFrame — needed to compute 5D % change.

    Returns
    -------
    dict
        Archive row with all 62 column keys. Uncomputable values are pd.NA.
    """
    from finance.apps.assistant._archive import COLUMN_ORDER

    row: dict = {col: pd.NA for col in COLUMN_ORDER}

    # --- Identity ---
    row["row_type"] = "market"
    row["symbol"] = symbol
    row["category"] = category
    # date is set by the caller (runner)

    # --- Price ---
    row["price"] = _safe_float(last.get("c"))

    # --- OHLC (market-only) ---
    row["open"] = _safe_float(last.get("o"))
    row["high"] = _safe_float(last.get("h"))
    row["low"] = _safe_float(last.get("l"))
    row["gap_pct"] = _safe_float(last.get("gappct"))

    # VIX and indices have no meaningful volume; store as NA when 0 or missing
    vol = _safe_float(last.get("v"))
    row["volume"] = vol if (vol is not pd.NA and vol > 0) else pd.NA

    # --- Daily change ---
    row["change_pct"] = _safe_float(last.get("pct"))

    # --- 5D change — computed from close series ---
    c_series = df_day["c"].dropna() if "c" in df_day.columns else pd.Series(dtype=float)
    if len(c_series) >= 6:
        c_now = c_series.iloc[-1]
        c_5d_ago = c_series.iloc[-6]
        row["change_5d_pct"] = _safe_float((c_now / c_5d_ago - 1) * 100)
    # else stays NA

    # --- Performance lookbacks from swing_indicators ---
    row["change_1m_pct"] = _safe_float(last.get("1M_chg"))
    row["change_3m_pct"] = _safe_float(last.get("3M_chg"))
    row["change_6m_pct"] = _safe_float(last.get("6M_chg"))
    row["change_52w_pct"] = _safe_float(last.get("12M_chg"))

    # --- Volume and volatility ---
    row["rvol_20d"] = _safe_float(last.get("rvol20"))
    row["atr_pct_20d"] = _safe_float(last.get("atrp20"))
    row["iv_percentile"] = _safe_float(last.get("iv_pct"))
    row["hv20"] = _safe_float(last.get("hv20"))
    row["iv_rank"] = _safe_float(last.get("iv_rank"))

    # --- SMA distance ---
    row["pct_from_50d_sma"] = _safe_float(last.get("ma50_dist"))

    # --- SMA slopes: normalize from price-units/day → %/day ---
    ma50_slope = _safe_float(last.get("ma50_slope"))
    ma50_val = _safe_float(last.get("ma50"))
    if ma50_slope is not pd.NA and ma50_val is not pd.NA and ma50_val != 0:
        row["slope_50d_sma"] = (ma50_slope / ma50_val) * 100
    # else stays NA

    ma200_slope = _safe_float(last.get("ma200_slope"))
    ma200_val = _safe_float(last.get("ma200"))
    if ma200_slope is not pd.NA and ma200_val is not pd.NA and ma200_val != 0:
        row["slope_200d_sma"] = (ma200_slope / ma200_val) * 100
    # else stays NA

    # --- BB% ---
    c = _safe_float(last.get("c"))
    bb_lower = _safe_float(last.get("bb_lower"))
    bb_upper = _safe_float(last.get("bb_upper"))
    if (
        c is not pd.NA
        and bb_lower is not pd.NA
        and bb_upper is not pd.NA
        and (bb_upper - bb_lower) != 0
    ):
        row["bb_pct"] = (c - bb_lower) / (bb_upper - bb_lower) * 100
    # else stays NA

    # --- TTM Squeeze ---
    squeeze_on = last.get("squeeze_on")
    if squeeze_on is None or (isinstance(squeeze_on, float) and pd.isna(squeeze_on)):
        row["ttm_squeeze"] = pd.NA
    elif squeeze_on:
        row["ttm_squeeze"] = "On"
    else:
        row["ttm_squeeze"] = "Off"

    return row


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_market_instrument(instr: dict, *, datasource: str = "ibkr") -> dict | None:
    """
    Load one market instrument and return an archive row dict.

    Uses SwingTradingData(stk_symbol, datasource='ibkr') to fetch live EOD
    data. Forex instruments use the `^` prefix convention (e.g. '^EURUSD').

    datasource='ibkr' fetches live bars from IBKR and reads split/financial
    metadata from local offline files — no MySQL/dolt connection required.

    Returns None on any failure:
    - SwingTradingData raises or returns empty df_day
    - price is 0 or missing (physically impossible for these instruments)

    The caller is responsible for logging/skipping None results.
    """
    symbol = instr["symbol"]
    stk_symbol = instr["stk_symbol"]
    category = instr["category"]

    try:
        data = SwingTradingData(stk_symbol, datasource=datasource)
    except Exception:
        log.error("Failed to load %s (%s): SwingTradingData raised", symbol, stk_symbol, exc_info=True)
        return None

    if getattr(data, "empty", False) or data.df_day is None or data.df_day.empty:
        log.error("Failed to load %s (%s): empty df_day", symbol, stk_symbol)
        return None

    last = data.df_day.iloc[-1]
    row = _swing_row_to_archive(symbol, category, last, data.df_day)

    # Price validation — 0 is never valid for any instrument we track
    price = row.get("price")
    if price is pd.NA or price == 0:
        log.warning(
            "Market instrument %s has invalid price (%s) — row omitted",
            symbol, price,
        )
        return None

    return row


def load_all_market(*, datasource: str = "ibkr") -> pd.DataFrame:
    """
    Load all 13 market instruments and return a combined archive DataFrame.

    Individual instrument failures are logged and skipped. The pipeline
    continues with the remaining instruments.

    Returns
    -------
    pd.DataFrame
        Market rows with all archive columns; row_type='market' for all.
        Returns an empty DataFrame (correct schema) if all instruments fail.
    """
    from finance.apps.assistant._archive import COLUMN_ORDER, DTYPES

    rows: list[dict] = []
    for instr in MARKET_INSTRUMENTS:
        row = load_market_instrument(instr, datasource=datasource)
        if row is not None:
            rows.append(row)
        else:
            log.warning("Skipping market instrument: %s", instr["symbol"])

    if not rows:
        df = pd.DataFrame(columns=COLUMN_ORDER)
        return df.astype(DTYPES)

    df = pd.DataFrame(rows).reindex(columns=COLUMN_ORDER)
    return df.astype(DTYPES)
