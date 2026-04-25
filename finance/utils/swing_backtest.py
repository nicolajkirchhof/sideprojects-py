"""
finance.utils.swing_backtest
==============================
Scoring backtest utilities for two workflows:

1. **Event scoring** — score historical earnings events from the momentum
   earnings dataset (all_YYYY.parquet) to validate that the weighted
   0–100 engine produces directionally correct signals on out-of-sample data.

2. **Trade scoring backtest** — replay historical closed trades through the
   scoring engine and compare scores with actual PnL outcomes.

Data sources:
  - Tradelog API (/api/trades/export) — closed trades with entry date + PnL
  - IBKR daily parquets (finance/_data/ibkr/) — historical OHLCV + volatility
  - Momentum earnings dataset (finance/_data/research/swing/momentum_earnings/)
  - SPY parquet — used to compute relative-strength fields

Usage::

    from datetime import date
    from finance.utils.swing_backtest import load_trade_entries, run_scoring_backtest

    entries = load_trade_entries()
    df = run_scoring_backtest(entries)
    print(df.describe())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from finance import utils
from finance.apps.analyst._config import load_config, TradelogConfig
from finance.apps.analyst._models import Candidate, EnrichedCandidate, TechnicalData
from finance.apps.analyst._tradelog import fetch_trades_for_review
from finance.apps.assistant._models import ScoringConfig
from finance.apps.assistant._scoring import score_candidate
from finance.apps.assistant._tags import assign_direction, assign_tags

log = logging.getLogger(__name__)

_SLOPE_FLAT_THRESHOLD = 0.05  # %/day — mirrors ScoringConfig default


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeEntry:
    """A single closed trade from the Tradelog."""
    trade_id: int
    symbol: str
    entry_date: date
    direction: str      # "long" | "short"
    pnl: float | None   # actual realised PnL in $


# ---------------------------------------------------------------------------
# Trade loading
# ---------------------------------------------------------------------------

def load_trade_entries(
    since: date | None = None,
    config: TradelogConfig | None = None,
) -> list[TradeEntry]:
    """
    Fetch closed trades from the Tradelog API and return as TradeEntry list.

    Parameters
    ----------
    since:
        Only return trades with entry date on or after this date.
    config:
        Tradelog connection config. Defaults to load_config().tradelog.

    Returns
    -------
    list[TradeEntry]
        One entry per closed trade. Trades with no parseable entry date are
        skipped with a warning.
    """
    cfg = config or load_config().tradelog
    raw = fetch_trades_for_review(cfg, status="Closed", since=since)

    entries: list[TradeEntry] = []
    for row in raw:
        trade_id = row.get("id") or row.get("tradeId") or row.get("trade_id")
        symbol = row.get("symbol") or row.get("Symbol")
        direction_raw = (
            row.get("directional") or row.get("direction") or row.get("Directional") or ""
        ).lower()
        pnl_raw = row.get("pnl") or row.get("pnL") or row.get("realizedPnl")

        # Parse entry date
        entry_date_raw = (
            row.get("entryDate") or row.get("entry_date") or row.get("date")
        )
        try:
            entry_date = pd.Timestamp(entry_date_raw).date()
        except Exception:
            log.warning("Skipping trade %s — unparseable entry date: %r", trade_id, entry_date_raw)
            continue

        if not symbol:
            log.warning("Skipping trade %s — missing symbol", trade_id)
            continue

        direction = "short" if "short" in direction_raw else "long"
        pnl = float(pnl_raw) if pnl_raw is not None else None

        entries.append(TradeEntry(
            trade_id=int(trade_id) if trade_id is not None else 0,
            symbol=str(symbol).upper(),
            entry_date=entry_date,
            direction=direction,
            pnl=pnl,
        ))

    return entries


# ---------------------------------------------------------------------------
# Candidate reconstruction
# ---------------------------------------------------------------------------

def _load_parquet(symbol: str) -> pd.DataFrame | None:
    """Load the IBKR daily parquet for a symbol. Returns None if not found."""
    path = utils.ibkr.cache_path(symbol)
    try:
        df = pd.read_parquet(path)
        df.sort_index(inplace=True)
        return df
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("Failed to read parquet for %s: %s", symbol, exc)
        return None


def _slope_category(slope_pct_per_day: float | None, threshold: float = _SLOPE_FLAT_THRESHOLD) -> str | None:
    if slope_pct_per_day is None or np.isnan(slope_pct_per_day):
        return None
    if abs(slope_pct_per_day) <= threshold:
        return "flat"
    return "rising" if slope_pct_per_day > 0 else "falling"


def _safe(val) -> float | None:
    """Return float or None for scalar series values."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def reconstruct_candidate(
    symbol: str,
    entry_date: date,
    spy_df: pd.DataFrame | None = None,
) -> EnrichedCandidate | None:
    """
    Build an EnrichedCandidate from IBKR parquet data at ``entry_date``.

    Mirrors what ``_market.py`` does for live data but slices the parquet to
    the entry date so indicators are computed on data available at that time.

    Parameters
    ----------
    symbol:
        Stock ticker (e.g. "AAPL").
    entry_date:
        The historical entry date to reconstruct.
    spy_df:
        Pre-loaded SPY parquet with swing_indicators already run.
        Pass once and reuse across calls for efficiency. If None, it is loaded
        inline (slower but correct).

    Returns
    -------
    EnrichedCandidate or None
        None when the symbol has no cached parquet or ``entry_date`` falls
        outside the available date range.
    """
    df_raw = _load_parquet(symbol)
    if df_raw is None or df_raw.empty:
        log.debug("No parquet for %s", symbol)
        return None

    # Slice to entry_date (inclusive) — simulate data visible on that day
    cutoff = pd.Timestamp(entry_date)
    df_slice = df_raw[df_raw.index <= cutoff]

    if len(df_slice) < 30:
        log.debug("%s: too few bars before %s (%d)", symbol, entry_date, len(df_slice))
        return None

    # Run swing_indicators on the slice
    try:
        df = utils.indicators.swing_indicators(df_slice.copy())
    except Exception as exc:
        log.warning("swing_indicators failed for %s @ %s: %s", symbol, entry_date, exc)
        return None

    last = df.iloc[-1]
    price = _safe(last.get("c"))
    if not price:
        return None

    # ---- Relative performance vs SPY ----
    perf_5d = perf_1m = perf_3m = None
    spy = spy_df
    if spy is None:
        spy_raw = _load_parquet("SPY")
        if spy_raw is not None and not spy_raw.empty:
            try:
                spy = utils.indicators.swing_indicators(spy_raw[spy_raw.index <= cutoff].copy())
            except Exception:
                spy = None

    if spy is not None and not spy.empty:
        spy_last = spy[spy.index <= cutoff]
        if not spy_last.empty:
            spy_row = spy_last.iloc[-1]
            stock_1m = _safe(last.get("1M_chg"))
            stock_3m = _safe(last.get("3M_chg"))
            stock_5d = _safe(last.get("pct"))  # approximate with 5d from parquet
            spy_1m = _safe(spy_row.get("1M_chg"))
            spy_3m = _safe(spy_row.get("3M_chg"))

            # 5d relative: use 5-bar return from parquet
            c_series = df["c"].dropna()
            if len(c_series) >= 6:
                ret_5d = (c_series.iloc[-1] / c_series.iloc[-6] - 1) * 100
                spy_c = spy_last["c"].dropna()
                if len(spy_c) >= 6:
                    spy_ret_5d = (spy_c.iloc[-1] / spy_c.iloc[-6] - 1) * 100
                    perf_5d = ret_5d - spy_ret_5d

            if stock_1m is not None and spy_1m is not None:
                perf_1m = stock_1m - spy_1m
            if stock_3m is not None and spy_3m is not None:
                perf_3m = stock_3m - spy_3m

    # ---- Build Candidate ----
    ma50 = _safe(last.get("ma50"))
    ma200 = _safe(last.get("ma200"))
    ma50_slope_raw = _safe(last.get("ma50_slope"))
    ma200_slope_raw = _safe(last.get("ma200_slope"))
    slope_50_pct = (ma50_slope_raw / ma50 * 100) if (ma50 and ma50_slope_raw is not None) else None
    slope_200_pct = (ma200_slope_raw / ma200 * 100) if (ma200 and ma200_slope_raw is not None) else None

    bb_lower = _safe(last.get("bb_lower"))
    bb_upper = _safe(last.get("bb_upper"))
    bb_pct = None
    if bb_lower is not None and bb_upper is not None and (bb_upper - bb_lower) != 0:
        bb_pct = (price - bb_lower) / (bb_upper - bb_lower) * 100

    squeeze_on_raw = last.get("squeeze_on")
    if squeeze_on_raw is None or (isinstance(squeeze_on_raw, float) and np.isnan(squeeze_on_raw)):
        ttm_squeeze = None
    elif squeeze_on_raw:
        ttm_squeeze = "On"
    else:
        ttm_squeeze = "Off"

    # 52W high/low from the slice
    high_52w = _safe(df["h"].tail(252).max()) if "h" in df.columns else None
    low_52w = _safe(df["l"].tail(252).min()) if "l" in df.columns else None
    high_52w_distance_pct = ((price - high_52w) / high_52w * 100) if high_52w else None

    candidate = Candidate(
        symbol=symbol,
        price=price,
        change_pct=_safe(last.get("pct")),
        change_5d_pct=_safe(
            (df["c"].iloc[-1] / df["c"].iloc[-6] - 1) * 100
            if len(df) >= 6 else None
        ),
        change_1m_pct=_safe(last.get("1M_chg")),
        change_3m_pct=_safe(last.get("3M_chg")),
        change_6m_pct=_safe(last.get("6M_chg")),
        change_52w_pct=_safe(last.get("12M_chg")),
        high_52w_distance_pct=high_52w_distance_pct,
        rvol_20d=_safe(last.get("rvol20")),
        atr_pct_20d=_safe(last.get("atrp20")),
        adr_pct_20d=_safe(last.get("atrp20")),    # best available proxy without scanner ADR
        pct_from_50d_sma=_safe(last.get("ma50_dist")),
        slope_50d_sma=slope_50_pct,
        slope_200d_sma=slope_200_pct,
        bb_pct=bb_pct,
        ttm_squeeze=ttm_squeeze,
        weighted_alpha=_safe(last.get("12M_chg")),  # proxy
        perf_vs_market_5d=perf_5d,
        perf_vs_market_1m=perf_1m,
        perf_vs_market_3m=perf_3m,
    )

    # ---- Build TechnicalData ----
    bb_width = (bb_upper - bb_lower) if (bb_upper and bb_lower) else None
    bb_width_avg_20: float | None = None
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        widths = (df["bb_upper"] - df["bb_lower"]).tail(20).dropna()
        bb_width_avg_20 = float(widths.mean()) if len(widths) >= 5 else None

    rs_slope_10d: float | None = None
    if spy is not None and not spy.empty and "c" in df.columns and "c" in spy.columns:
        sym_c = df["c"].dropna()
        spy_c2 = spy["c"].reindex(sym_c.index, method="ffill").dropna()
        if len(sym_c) >= 11 and len(spy_c2) >= 11:
            rs_line = (sym_c / spy_c2).tail(11)
            rs_slope_10d = float((rs_line.iloc[-1] - rs_line.iloc[0]) / 10)

    technicals = TechnicalData(
        sma_5=_safe(last.get("ma5")),
        sma_10=_safe(last.get("ma10")),
        sma_20=_safe(last.get("ma20")),
        sma_50=ma50,
        sma_200=ma200,
        sma_50_slope=_slope_category(slope_50_pct),
        sma_200_slope=_slope_category(slope_200_pct),
        bb_width=bb_width,
        bb_width_avg_20=bb_width_avg_20,
        atr_14=_safe(last.get("atr14")),
        high_52w=high_52w,
        low_52w=low_52w,
        return_12m=_safe(last.get("12M_chg")),
        rs_slope_10d=rs_slope_10d,
        rvol=_safe(last.get("rvol20")),
        volume_contracting=None,   # not derivable from daily parquet alone
    )

    return EnrichedCandidate(
        candidate=candidate,
        technicals=technicals,
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Event-row reconstruction (momentum earnings dataset)
# ---------------------------------------------------------------------------

def reconstruct_candidate_from_event_row(
    row: pd.Series,
    direction: str = "long",
) -> EnrichedCandidate | None:
    """
    Map a momentum earnings dataset row (T=0 snapshot) to an EnrichedCandidate.

    Columns consumed from the dataset:
      c0              → price (close at event day)
      1M_chg          → change_1m_pct
      3M_chg          → change_3m_pct
      6M_chg          → change_6m_pct
      12M_chg         → change_52w_pct
      ma50_dist0      → pct_from_50d_sma (already %)
      ma50_slope0     → slope_50d_sma (%/day, converted from price-units/day)
      ma200_slope0    → slope_200d_sma (%/day)
      atrp200         → atr_pct_20d / adr_pct_20d proxy (ATR20 as % of price)
      rvol200         → rvol_20d
      sue             → earnings_surprise_pct (SUE as directional proxy)
      spy0, spy-21    → SPY context for 1M relative perf
      spy0, spy-60    → SPY context for 3M relative perf

    Parameters
    ----------
    row:
        A single row (pd.Series) from the momentum earnings parquet.
    direction:
        Unused here but preserved for API symmetry with reconstruct_candidate.

    Returns
    -------
    EnrichedCandidate or None
        None when ``c0`` is missing or zero.
    """
    price = _safe(row.get("c0"))
    if not price:
        return None

    # Slope: price-units/day → %/day
    ma50_slope_raw = _safe(row.get("ma50_slope0"))
    ma200_slope_raw = _safe(row.get("ma200_slope0"))
    slope_50_pct = (ma50_slope_raw / (price * 0.01)) if ma50_slope_raw is not None else None
    slope_200_pct = (ma200_slope_raw / (price * 0.01)) if ma200_slope_raw is not None else None

    # Relative performance vs SPY
    # SPY columns are cumulative % returns from a common base date.
    # (spy0 - spy-N) ≈ SPY N-day return ending at event day.
    stock_1m = _safe(row.get("1M_chg"))
    stock_3m = _safe(row.get("3M_chg"))
    spy0 = _safe(row.get("spy0"))
    spy_m21 = _safe(row.get("spy-21"))
    spy_m60 = _safe(row.get("spy-60"))

    perf_1m: float | None = None
    perf_3m: float | None = None
    if stock_1m is not None and spy0 is not None and spy_m21 is not None:
        perf_1m = stock_1m - (spy0 - spy_m21)
    if stock_3m is not None and spy0 is not None and spy_m60 is not None:
        perf_3m = stock_3m - (spy0 - spy_m60)

    # SMA levels back-computed from price + % distance
    ma50_dist = _safe(row.get("ma50_dist0"))
    ma200_dist = _safe(row.get("ma200_dist0"))
    sma_50 = (price / (1 + ma50_dist / 100)) if ma50_dist is not None else None
    sma_200 = (price / (1 + ma200_dist / 100)) if ma200_dist is not None else None

    atrp20 = _safe(row.get("atrp200"))   # ATR20 as % at T=0
    atrp14 = _safe(row.get("atrp140"))   # ATR14 as % at T=0
    rvol20 = _safe(row.get("rvol200"))   # RVOL20 at T=0

    candidate = Candidate(
        symbol=str(row.get("symbol", "")),
        price=price,
        change_1m_pct=stock_1m,
        change_3m_pct=stock_3m,
        change_6m_pct=_safe(row.get("6M_chg")),
        change_52w_pct=_safe(row.get("12M_chg")),
        pct_from_50d_sma=ma50_dist,
        slope_50d_sma=slope_50_pct,
        slope_200d_sma=slope_200_pct,
        atr_pct_20d=atrp20,
        adr_pct_20d=atrp20,
        rvol_20d=rvol20,
        earnings_surprise_pct=_safe(row.get("sue")),
        perf_vs_market_1m=perf_1m,
        perf_vs_market_3m=perf_3m,
    )

    technicals = TechnicalData(
        sma_50=sma_50,
        sma_200=sma_200,
        sma_50_slope=_slope_category(slope_50_pct),
        sma_200_slope=_slope_category(slope_200_pct),
        bb_width=None,
        bb_width_avg_20=None,
        atr_14=(atrp14 * price / 100) if atrp14 is not None else None,
        return_12m=_safe(row.get("12M_chg")),
        rvol=rvol20,
    )

    return EnrichedCandidate(
        candidate=candidate,
        technicals=technicals,
        data_available=True,
    )


def run_event_scoring(
    events_df: pd.DataFrame,
    fwd_col: str = "cpct10",
    direction: str = "long",
    direction_col: str | None = None,
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    """
    Score each row in an events DataFrame and return one row per event.

    Intended for the momentum earnings dataset (all_YYYY.parquet).

    Parameters
    ----------
    events_df:
        DataFrame of events, one row per event. Must contain the columns
        consumed by ``reconstruct_candidate_from_event_row`` plus ``fwd_col``.
    fwd_col:
        Name of the forward-return column to use as the outcome variable.
        Default ``"cpct10"`` (10-day forward return in %).
    direction:
        Default trade direction when ``direction_col`` is absent or has no
        value. ``"long"`` or ``"short"``.
    direction_col:
        Optional column name in ``events_df`` that supplies a per-row
        direction (``"long"`` or ``"short"``). When set, overrides
        ``direction`` for each row individually. Falls back to ``direction``
        when the column value is missing or blank.
    config:
        Scoring weights. Defaults to standard 25/25/15/20/15 split.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, date, direction, score_total, score_d1..d5,
        fwd_return, win, spy_regime.
        Rows where ``c0`` is missing/zero or scoring fails are silently dropped.
    """
    if config is None:
        config = ScoringConfig(
            weights={1: 25, 2: 25, 3: 15, 4: 20, 5: 15},
            tag_bonus_per_tag=2,
            tag_bonus_cap=12,
        )

    rows: list[dict] = []
    for _, row in events_df.iterrows():
        # Resolve per-row direction
        if direction_col:
            row_dir = str(row.get(direction_col) or direction).strip().lower()
            if row_dir not in ("long", "short"):
                row_dir = direction
        else:
            row_dir = direction

        ec = reconstruct_candidate_from_event_row(row, row_dir)
        if ec is None:
            continue

        fwd_return = _safe(row.get(fwd_col))

        # Regime proxy: SPY up over prior 50 days (cumulative return columns)
        spy0 = _safe(row.get("spy0"))
        spy_m50 = _safe(row.get("spy-50"))
        spy_regime: str | None
        if spy0 is not None and spy_m50 is not None:
            spy_regime = "go" if spy_m50 < spy0 else "no-go"
        else:
            spy_regime = None

        tags = assign_tags(ec.candidate, scanner_sets={})
        try:
            result = score_candidate(ec, row_dir, tags, config)
        except Exception as exc:
            log.warning("Scoring failed for %s: %s", ec.candidate.symbol, exc)
            continue

        # win: for long, positive fwd return; for short, negative fwd return
        win: bool | None
        if fwd_return is not None:
            win = fwd_return > 0 if row_dir == "long" else fwd_return < 0
        else:
            win = None

        row_out: dict = {
            "symbol": ec.candidate.symbol,
            "date": row.get("date"),
            "direction": row_dir,
            "score_total": result.total,
            "fwd_return": fwd_return,
            "win": win,
            "spy_regime": spy_regime,
        }
        for dim in result.dimensions:
            row_out[f"score_d{dim.dimension}"] = dim.weighted_score

        rows.append(row_out)

    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "date", "direction", "score_total",
            "score_d1", "score_d2", "score_d3", "score_d4", "score_d5",
            "fwd_return", "win", "spy_regime",
        ])

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_scoring_backtest(
    entries: list[TradeEntry],
    config: ScoringConfig | None = None,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Score each historical trade entry and join with actual outcome.

    Parameters
    ----------
    entries:
        Trade entries from ``load_trade_entries()``.
    config:
        Scoring weights. Defaults to standard 25/25/15/20/15 split.
    spy_df:
        Pre-loaded SPY daily parquet with swing_indicators. Pass once
        to avoid reloading on every call. Loaded lazily if None.

    Returns
    -------
    pd.DataFrame
        One row per trade, columns:
          trade_id, symbol, entry_date, direction, pnl, win,
          score_total, score_d1..d5, score_direction
    """
    if config is None:
        config = ScoringConfig(
            weights={1: 25, 2: 25, 3: 15, 4: 20, 5: 15},
            tag_bonus_per_tag=2,
            tag_bonus_cap=12,
        )

    # Load SPY once for relative-strength computation
    if spy_df is None:
        spy_raw = _load_parquet("SPY")
        if spy_raw is not None and not spy_raw.empty:
            try:
                spy_df = utils.indicators.swing_indicators(spy_raw.copy())
            except Exception as exc:
                log.warning("Could not load SPY for RS computation: %s", exc)
                spy_df = None

    rows: list[dict] = []
    for entry in entries:
        ec = reconstruct_candidate(entry.symbol, entry.entry_date, spy_df=spy_df)
        if ec is None:
            log.debug("Skipping %s @ %s — no data", entry.symbol, entry.entry_date)
            continue

        tags = assign_tags(ec.candidate, scanner_sets={})
        inferred_direction = assign_direction(tags)
        direction = entry.direction  # use actual trade direction, not inferred

        try:
            result = score_candidate(ec, direction, tags, config)
        except Exception as exc:
            log.warning("Scoring failed for %s @ %s: %s", entry.symbol, entry.entry_date, exc)
            continue

        row: dict = {
            "trade_id": entry.trade_id,
            "symbol": entry.symbol,
            "entry_date": entry.entry_date,
            "direction": direction,
            "score_direction": inferred_direction,
            "pnl": entry.pnl,
            "win": (entry.pnl > 0) if entry.pnl is not None else None,
            "score_total": result.total,
            "score_tag_bonus": result.tag_bonus,
        }
        for dim in result.dimensions:
            row[f"score_d{dim.dimension}"] = dim.weighted_score

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "trade_id", "symbol", "entry_date", "direction", "score_direction",
            "pnl", "win", "score_total", "score_tag_bonus",
            "score_d1", "score_d2", "score_d3", "score_d4", "score_d5",
        ])

    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
