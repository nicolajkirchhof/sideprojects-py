"""
build_vwap_extrema_parquet.py
=============================
Builds per-symbol vwap_extrema Parquet files used by BT-6-S2.

Two-step process:
  Step A — Consolidate existing per-day PKL files (swing_vwap_ad/) into a single
            Parquet file, extracting only the columns needed by the backtest.
  Step B — Generate new rows for dates not covered by the PKLs, using the 5-min
            Parquet intraday store.  Reuses the same indicators.py logic that
            produced the original PKLs.

Output: finance/_data/intraday/vwap_extrema/{SYMBOL}.parquet

Run from repo root:
    python finance/intraday_pm/data/build_vwap_extrema_parquet.py
"""
from __future__ import annotations

import sys
import glob
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from finance.utils.intraday import get_bars
from finance.utils import indicators

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PKL_ROOT = "N:/My Drive/Trading/Strategies/swing_vwap_ad"
OUT_DIR = Path("finance/_data/intraday/vwap_extrema")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Symbols present in PKL store (Step A available)
SYMBOLS_PKL = ["IBDE40", "IBGB100", "IBUS500", "IBUST100", "IBJP225", "IBES35"]

# All symbols to build (Parquet-only symbols get Step B from 2020-01-01)
SYMBOLS = SYMBOLS_PKL + ["IBEU50", "IBFR40", "IBCH20", "IBNL25", "IBUS30", "IBAU200", "USGOLD"]

SCHEMA = "cfd"
UTC = timezone.utc

PARQUET_ONLY_START = date(2020, 1, 1)  # Step B start date for symbols with no PKL history

# Session open/close hours in UTC for each symbol (for overnight/session slicing)
SESSION_UTC: dict[str, tuple[int, int]] = {
    "IBDE40":   (7, 17),   # 09:00–17:30 Frankfurt ≈ 07:00–15:30 UTC (winter)
    "IBGB100":  (7, 17),   # 08:00–16:30 London ≈ 07:00–15:30 UTC
    "IBUS500":  (14, 21),  # 09:30–16:00 ET = 14:30–21:00 UTC
    "IBUST100": (14, 21),
    "IBJP225":  (0, 6),    # 09:00–15:30 Tokyo ≈ 00:00–06:30 UTC
    "IBES35":   (7, 17),
    "IBEU50":   (7, 17),   # EuroStoxx 50 — Frankfurt session
    "IBFR40":   (7, 17),   # CAC 40 — Paris session
    "IBCH20":   (7, 17),   # SMI — Zurich session
    "IBNL25":   (7, 17),   # AEX — Amsterdam session
    "IBUS30":   (14, 21),  # Dow Jones — same ET session as IBUS500
    "IBAU200":  (0, 6),    # ASX 200 — Sydney 10:00–16:00 ≈ 00:00–06:00 UTC
    "USGOLD":   (7, 17),   # Gold — London session anchor
}

PULLBACK_THRESHOLD_MULTIPLIER = 0.3
MIN_BARS_PER_DAY = 30

# ---------------------------------------------------------------------------
# Schema: columns to keep in Parquet
# ---------------------------------------------------------------------------
KEEP_COLS = [
    "trade_date",
    "symbol",
    "ts",
    "candle_sentiment",
    "candle_atr_pts",
    "in_trend",
    "pts_move",
    "sl_pts_offset",
    "bracket_trigger",
    "bracket_entry_in_trend",
    "data_source",   # "pkl" or "parquet"
]


# ---------------------------------------------------------------------------
# Step A: Consolidate PKLs
# ---------------------------------------------------------------------------
def _process_bracket_moves(bm: pd.DataFrame, trade_date: date, symbol: str, source: str) -> pd.DataFrame:
    """Add derived columns and normalise a df_bracket_moves DataFrame."""
    bm = bm.copy()
    bm["trade_date"] = trade_date
    bm["symbol"] = symbol
    bm["data_source"] = source

    # Derive bracket_entry_in_trend
    in_trend_int = bm["in_trend"].astype(int).replace(0, -1)
    bm["trend_sentiment"] = bm["candle_sentiment"] * in_trend_int
    bm["bracket_entry_in_trend"] = (bm["bracket_trigger"] == bm["trend_sentiment"])

    # Ensure ts column is present (PKLs use a 'ts' column before set_index)
    if "ts" not in bm.columns:
        bm = bm.reset_index().rename(columns={"index": "ts"})

    return bm[KEEP_COLS]


def consolidate_pkls(symbol: str) -> pd.DataFrame:
    """Read all per-day PKLs for a symbol and return a combined DataFrame."""
    pattern = f"{PKL_ROOT}/{symbol}/{symbol}_*.pkl"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No PKL files found for {symbol}")
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for fpath in files:
        fname = Path(fpath).stem            # e.g. IBDE40_2020-01-03
        day_str = fname.split("_", 1)[1]    # 2020-01-03
        trade_date = date.fromisoformat(day_str)

        try:
            with open(fpath, "rb") as fh:
                d = pickle.load(fh)
            bm = d.get("df_bracket_moves")
            if bm is None or bm.empty:
                continue
            rows.append(_process_bracket_moves(bm, trade_date, symbol, "pkl"))
        except Exception as exc:
            print(f"  Warning: skipping {fpath}: {exc}")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Step B: Generate new rows from Parquet data
# ---------------------------------------------------------------------------
@dataclass
class DayRef:
    """Reference levels for a single trading day, mirroring TradingDayData attributes."""
    cdh: float
    cdl: float
    pdh: float
    pdl: float
    cwh: float
    cwl: float
    pwh: float
    pwl: float
    onh: float
    onl: float


def _build_day_ref(trade_date: date, df5_all: pd.DataFrame, session_utc_range: tuple[int, int]) -> DayRef:
    """
    Compute DayRef reference levels from the 5-min DataFrame for one trading day.
    df5_all should cover at least prior_week to next_day.
    """
    open_h, close_h = session_utc_range

    # Current day session bars
    day_mask = (
        (df5_all.index.date == trade_date) &
        (df5_all.index.hour >= open_h) &
        (df5_all.index.hour < close_h)
    )
    day_bars = df5_all[day_mask]

    # Prior day (previous calendar day that has bars)
    prior_dates = [d for d in sorted(df5_all.index.date) if d < trade_date]
    prior_date = prior_dates[-1] if prior_dates else None
    prior_mask = (
        (df5_all.index.date == prior_date) &
        (df5_all.index.hour >= open_h) &
        (df5_all.index.hour < close_h)
    ) if prior_date else pd.Series(False, index=df5_all.index)
    prior_bars = df5_all[prior_mask]

    # Current week (Mon–Sun containing trade_date)
    trade_dt = pd.Timestamp(trade_date)
    week_start = trade_dt - pd.Timedelta(days=trade_dt.dayofweek)  # Monday
    cw_mask = (df5_all.index.date >= week_start.date()) & (df5_all.index.date < trade_date)
    cw_bars = df5_all[cw_mask]

    # Prior week
    pw_end = week_start
    pw_start = pw_end - pd.Timedelta(days=7)
    pw_mask = (df5_all.index.date >= pw_start.date()) & (df5_all.index.date < pw_end.date())
    pw_bars = df5_all[pw_mask]

    # Overnight: bars on trade_date before session open
    on_mask = (df5_all.index.date == trade_date) & (df5_all.index.hour < open_h)
    on_bars = df5_all[on_mask]

    def _h(b: pd.DataFrame) -> float:
        return float(b["high"].max()) if not b.empty else np.nan

    def _l(b: pd.DataFrame) -> float:
        return float(b["low"].min()) if not b.empty else np.nan

    return DayRef(
        cdh=_h(day_bars),  cdl=_l(day_bars),
        pdh=_h(prior_bars), pdl=_l(prior_bars),
        cwh=_h(cw_bars),   cwl=_l(cw_bars),
        pwh=_h(pw_bars),   pwl=_l(pw_bars),
        onh=_h(on_bars),   onl=_l(on_bars),
    )


def _prepare_df5(day_bars: pd.DataFrame) -> pd.DataFrame:
    """Add VWAP3, 20EMA, lh, oc columns expected by indicators.trading_day_moves."""
    df = day_bars.copy()
    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})
    df["VWAP3"] = (df["h"] + df["l"] + df["c"]) / 3
    df["20EMA"] = df["VWAP3"].ewm(span=20, adjust=False).mean()
    df["lh"] = df["h"] - df["l"]
    df["oc"] = df["c"] - df["o"]
    df.loc[df["lh"] == 0, "lh"] = 0.001
    df.loc[df["oc"] == 0, "oc"] = 0.001
    return df


def _cancel_moves_stats(candle: pd.Series, next_extrema: pd.Series, df_5m: pd.DataFrame) -> dict:
    """Replicate cancel_moves_stats() from the original generation script."""
    candle_sentiment = 1 if candle["c"] - candle["o"] >= 0 else -1
    candle_atr_pts = candle["h"] - candle["l"]
    move_sentiment = 1 if next_extrema["value"] - candle["VWAP3"] >= 0 else -1
    in_trend = candle_sentiment == move_sentiment
    pts_move = abs(candle["VWAP3"] - next_extrema["value"])

    candles_move = df_5m[(df_5m.index > candle.name) & (df_5m.index <= next_extrema["ts"])]
    bull_trigger = candles_move[candles_move["h"] > candle["h"]]
    bear_trigger = candles_move[candles_move["l"] < candle["l"]]

    if bull_trigger.empty and bear_trigger.empty:
        bracket_trigger = 0
    elif bull_trigger.empty:
        bracket_trigger = -1
    elif bear_trigger.empty:
        bracket_trigger = 1
    elif bull_trigger.index[0] > bear_trigger.index[0]:
        bracket_trigger = -1
    elif bull_trigger.index[0] < bear_trigger.index[0]:
        bracket_trigger = 1
    else:
        bracket_trigger = move_sentiment * -1

    if bracket_trigger > 0:
        bm_candles = candles_move[candles_move.index >= bull_trigger.index[0]]
    elif bracket_trigger < 0:
        bm_candles = candles_move[candles_move.index >= bear_trigger.index[0]]
    else:
        bm_candles = None

    sl_value = (
        candles_move["l"].min() if move_sentiment > 0 else candles_move["h"].max()
    )
    sl_pts_offset = (
        candle["l"] - sl_value if move_sentiment > 0 else sl_value - candle["h"]
    )

    return {
        "ts": candle.name,
        "candle_sentiment": candle_sentiment,
        "candle_atr_pts": candle_atr_pts,
        "in_trend": in_trend,
        "pts_move": pts_move,
        "sl_pts_offset": sl_pts_offset,
        "bracket_trigger": bracket_trigger,
    }


def generate_from_parquet(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Generate bracket move rows for [start_date, end_date) using Parquet 5-min data.
    """
    session_utc = SESSION_UTC.get(symbol, (7, 21))
    open_h, close_h = session_utc

    # Load 5-min data with a 3-week lookback for reference levels
    load_start = datetime.combine(
        start_date - timedelta(days=21), datetime.min.time(), tzinfo=UTC
    )
    load_end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)

    df5_raw = get_bars(symbol, SCHEMA, load_start, load_end, period="5min")
    if df5_raw.empty:
        print(f"  No Parquet data for {symbol}")
        return pd.DataFrame()

    df5_raw = df5_raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})

    all_rows: list[pd.DataFrame] = []
    trading_dates = sorted(set(
        d for d in df5_raw.index.date
        if start_date <= d < end_date
    ))

    for trade_date in trading_dates:
        # Session bars for this day
        day_mask = (
            (df5_raw.index.date == trade_date) &
            (df5_raw.index.hour >= open_h) &
            (df5_raw.index.hour < close_h)
        )
        day_bars_raw = df5_raw[day_mask]
        if len(day_bars_raw) < MIN_BARS_PER_DAY:
            continue

        df5 = _prepare_df5(day_bars_raw)
        day_ref = _build_day_ref(trade_date, df5_raw, session_utc)

        try:
            df_extrema, _ = indicators.trading_day_moves(day_ref, df5, PULLBACK_THRESHOLD_MULTIPLIER)
        except Exception as exc:
            print(f"  Warning: extrema failed for {symbol} {trade_date}: {exc}")
            continue

        if len(df_extrema) < 2:
            continue

        bracket_rows: list[dict] = []
        OFFSET = 10  # skip last N bars (no next extrema available)
        for i in range(len(df5) - OFFSET):
            candle = df5.iloc[i]
            next_extremas = df_extrema[df_extrema["ts"] > candle.name]
            if next_extremas.empty:
                continue
            next_extrema = next_extremas.iloc[0]
            bracket_rows.append(_cancel_moves_stats(candle, next_extrema, df5))

        if not bracket_rows:
            continue

        bm = pd.DataFrame(bracket_rows)
        processed = _process_bracket_moves(bm, trade_date, symbol, "parquet")
        all_rows.append(processed)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_symbol(symbol: str) -> None:
    out_path = OUT_DIR / f"{symbol}.parquet"

    # --- Step A: consolidate PKLs (only for symbols that have a PKL history) ---
    df_pkl = pd.DataFrame()
    if symbol in SYMBOLS_PKL:
        print(f"  [A] Consolidating PKLs...")
        df_pkl = consolidate_pkls(symbol)
        if df_pkl.empty:
            print(f"  [A] No PKL data found")
        else:
            pkl_dates = set(df_pkl["trade_date"])
            last_pkl_date = max(pkl_dates)
            print(f"  [A] {len(df_pkl):,} rows, {len(pkl_dates)} days, last={last_pkl_date}")

    # --- Step B: generate from Parquet ---
    if df_pkl.empty:
        # No PKL history — generate everything from Parquet
        gap_start = PARQUET_ONLY_START
        print(f"  [B] Generating full history {gap_start} -> today from Parquet...")
    else:
        gap_start = max(df_pkl["trade_date"]) + timedelta(days=1)
        print(f"  [B] Generating gap {gap_start} -> today from Parquet...")

    gap_end = date.today()

    if gap_start < gap_end:
        df_new = generate_from_parquet(symbol, gap_start, gap_end)
        if not df_new.empty:
            print(f"  [B] Generated {len(df_new):,} new rows ({df_new['trade_date'].nunique()} days)")
            df_all = pd.concat([df_pkl, df_new], ignore_index=True) if not df_pkl.empty else df_new
        else:
            print(f"  [B] No new rows generated")
            df_all = df_pkl
    else:
        print(f"  [B] No gap to fill")
        df_all = df_pkl

    if df_all.empty:
        print(f"  No data produced — skipping write")
        return

    # Write Parquet
    df_all = df_all.sort_values(["trade_date", "ts"]).reset_index(drop=True)
    df_all.to_parquet(out_path, index=False)
    print(f"  Written: {out_path} ({len(df_all):,} rows, {df_all['trade_date'].nunique()} days)")


def main() -> None:
    for symbol in SYMBOLS:
        print(f"\n{symbol}")
        build_symbol(symbol)


if __name__ == "__main__":
    main()
