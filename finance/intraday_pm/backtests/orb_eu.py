"""
orb_eu.py
=========
BT-5-S2: Opening Range Breakout -- European session (IBDE40 + IBGB100).

Signal: first 15-min or 30-min candle at the local session open.
  IBDE40: 09:00 Frankfurt (Europe/Berlin)
  IBGB100: 08:00 London (Europe/London)

Entry: stop order 1 pt beyond the ORB high (long) or ORB low (short).
Stop: opposite ORB side (1 pt inside).
Target: 2x ORB range (2R).
Exit: 2R hit, stop hit, or session close (local time).
Cost: 2 pts round-trip spread.

Both long and short sides tested independently on every trading day.
Segmented by: day-of-week, gap open category, realised-ATR regime.

Run from repo root:
    python finance/intraday_pm/backtests/orb_eu.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from finance.utils.intraday import get_bars
from finance.intraday_pm.backtests.hougaard_dax import _fmt, _fmt_plain
from finance.intraday_pm.backtests.orb_us import _metrics_orb, _overall_row, _seg_table

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOLS = ["IBDE40", "IBGB100"]
SCHEMA = "cfd"
UTC = timezone.utc
START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2026, 4, 1, tzinfo=UTC)

TZ_MAP = {
    "IBDE40":  ZoneInfo("Europe/Berlin"),
    "IBGB100": ZoneInfo("Europe/London"),
}
SESSION_OPEN = {
    "IBDE40":  (9,  0),   # 09:00 Frankfurt
    "IBGB100": (8,  0),   # 08:00 London
}
SESSION_CLOSE = {
    "IBDE40":  (17, 30),  # 17:30 Frankfurt
    "IBGB100": (16, 30),  # 16:30 London
}

ORB_WINDOWS_MIN = [15, 30]
ENTRY_OFFSET_PTS = 1.0   # 1 pt for EU index CFDs
SPREAD_COST_PTS  = 2.0
TARGET_R         = 2.0
ATR_PERIOD       = 14
GAP_THRESHOLD_PCT = 0.2

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
GAP_LABELS    = ["gap_down", "flat", "gap_up"]
ATR_LABELS    = ["low_vol", "normal_vol", "high_vol"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_5min(symbol: str) -> pd.DataFrame:
    """Load 5-min bars and convert index to local session timezone."""
    tz = TZ_MAP[symbol]
    df = get_bars(symbol, SCHEMA, START, END, period="5min")
    if df.empty:
        raise RuntimeError(f"No 5-min data for {symbol}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(tz)
    return df[["open", "high", "low", "close"]]


def _load_daily(symbol: str) -> pd.DataFrame:
    df = get_bars(symbol, SCHEMA, START, END, period="1D")
    if df.empty:
        raise RuntimeError(f"No daily data for {symbol}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    return df[["open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# ATR and gap helpers
# ---------------------------------------------------------------------------
def _daily_atr(daily: pd.DataFrame) -> pd.Series:
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - daily["close"].shift(1)).abs(),
        (daily["low"]  - daily["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()
    atr.index = atr.index.date
    return atr


def _session_opens(df5: pd.DataFrame, symbol: str) -> pd.Series:
    """First 5-min bar open at the session open for each trading date."""
    open_h, open_m = SESSION_OPEN[symbol]
    mask = (df5.index.hour == open_h) & (df5.index.minute == open_m)
    opens = df5[mask]["open"].copy()
    opens.index = opens.index.date
    return opens


def _gap_series(daily_close: pd.Series, session_opens: pd.Series) -> pd.Series:
    dc = daily_close.copy()
    if hasattr(dc.index, "date"):
        dc.index = dc.index.date
    prev_close = dc.shift(1)
    common = session_opens.index.intersection(prev_close.index)
    return (session_opens.loc[common] / prev_close.loc[common] - 1) * 100


def _classify_gap(gap_pct: float) -> str:
    if pd.isna(gap_pct):
        return "flat"
    if gap_pct >= GAP_THRESHOLD_PCT:
        return "gap_up"
    if gap_pct <= -GAP_THRESHOLD_PCT:
        return "gap_down"
    return "flat"


def _classify_atr(atr_val: float, p33: float, p67: float) -> str:
    if pd.isna(atr_val):
        return "normal_vol"
    if atr_val <= p33:
        return "low_vol"
    if atr_val >= p67:
        return "high_vol"
    return "normal_vol"


# ---------------------------------------------------------------------------
# ORB simulation (EU-specific entry offset)
# ---------------------------------------------------------------------------
def _simulate_orb(
    direction: str,
    orb_high: float,
    orb_low: float,
    remaining: pd.DataFrame,
) -> dict:
    """
    Simulate one direction of an ORB trade on remaining 5-min bars (EU version).

    Long:  buy-stop at orb_high + offset; stop at orb_low - offset; target = entry + 2*range.
    Short: sell-stop at orb_low - offset; stop at orb_high + offset; target = entry - 2*range.

    Same-bar ambiguity (stop and target both hit): stop triggers first.
    """
    orb_range = orb_high - orb_low
    base = {"filled": False, "result_pts": np.nan, "win": np.nan, "orb_range": orb_range}

    if orb_range <= 0 or remaining.empty:
        return base

    if direction == "long":
        entry  = orb_high + ENTRY_OFFSET_PTS
        stop   = orb_low  - ENTRY_OFFSET_PTS
        target = entry + TARGET_R * orb_range
    else:
        entry  = orb_low  - ENTRY_OFFSET_PTS
        stop   = orb_high + ENTRY_OFFSET_PTS
        target = entry - TARGET_R * orb_range

    # Find fill bar
    filled_idx = None
    for idx, bar in remaining.iterrows():
        if direction == "long" and bar["high"] >= entry:
            filled_idx = idx
            break
        if direction == "short" and bar["low"] <= entry:
            filled_idx = idx
            break

    if filled_idx is None:
        return base

    base["filled"] = True
    post = remaining.loc[remaining.index > filled_idx]

    # Scan for target or stop (stop takes priority on same bar)
    for _, bar in post.iterrows():
        if direction == "long":
            if bar["low"]  <= stop:
                base["result_pts"] = (stop   - entry) - SPREAD_COST_PTS
                base["win"] = False
                return base
            if bar["high"] >= target:
                base["result_pts"] = (target - entry) - SPREAD_COST_PTS
                base["win"] = True
                return base
        else:
            if bar["high"] >= stop:
                base["result_pts"] = (entry - stop)   - SPREAD_COST_PTS
                base["win"] = False
                return base
            if bar["low"]  <= target:
                base["result_pts"] = (entry - target) - SPREAD_COST_PTS
                base["win"] = True
                return base

    # Session close exit
    if not post.empty:
        close_price = post.iloc[-1]["close"]
        if direction == "long":
            base["result_pts"] = (close_price - entry) - SPREAD_COST_PTS
        else:
            base["result_pts"] = (entry - close_price) - SPREAD_COST_PTS
        base["win"] = base["result_pts"] > 0

    return base


# ---------------------------------------------------------------------------
# Day loop
# ---------------------------------------------------------------------------
def run_orb(
    symbol: str,
    df5: pd.DataFrame,
    atr_series: pd.Series,
    gap_series: pd.Series,
    orb_window_min: int,
) -> pd.DataFrame:
    """Simulate ORB (both directions) for every session day."""
    open_h,  open_m  = SESSION_OPEN[symbol]
    close_h, close_m = SESSION_CLOSE[symbol]

    atr_valid = atr_series.dropna()
    atr_p33   = atr_valid.quantile(0.33)
    atr_p67   = atr_valid.quantile(0.67)

    n_orb_bars = orb_window_min // 5
    records: list[dict] = []
    df5 = df5.copy()
    df5["date"] = df5.index.date

    for trade_date, day_df in df5.groupby("date"):
        # Session mask (local time)
        session = day_df[
            (
                (day_df.index.hour == open_h) & (day_df.index.minute >= open_m)
            ) | (day_df.index.hour > open_h)
        ]
        session = session[
            (session.index.hour < close_h) |
            ((session.index.hour == close_h) & (session.index.minute <= close_m))
        ]

        if len(session) < n_orb_bars + 1:
            continue

        orb_bars  = session.iloc[:n_orb_bars]
        remaining = session.iloc[n_orb_bars:]

        orb_high = orb_bars["high"].max()
        orb_low  = orb_bars["low"].min()

        atr_val  = atr_series.get(trade_date, np.nan)
        gap_val  = gap_series.get(trade_date, np.nan)
        weekday  = pd.Timestamp(trade_date).day_name()
        gap_cat  = _classify_gap(gap_val)
        atr_cat  = _classify_atr(atr_val, atr_p33, atr_p67)

        for direction in ["long", "short"]:
            rec = _simulate_orb(direction, orb_high, orb_low, remaining)
            rec["date"]      = trade_date
            rec["symbol"]    = symbol
            rec["direction"] = direction
            rec["weekday"]   = weekday
            rec["gap_cat"]   = gap_cat
            rec["atr_cat"]   = atr_cat
            records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------
def build_section(results: dict[str, dict[int, pd.DataFrame]]) -> str:
    lines: list[str] = [
        "",
        "---",
        "",
        "## Opening Range Breakout -- European Session (BT-5-S2)",
        "",
        f"Generated: {date.today().isoformat()}  ",
        "Instruments: IBDE40 (Frankfurt), IBGB100 (London)  ",
        f"Period: {START.date().isoformat()} to {END.date().isoformat()}  ",
        "IBDE40 session: 09:00-17:30 Frankfurt  ",
        "IBGB100 session: 08:00-16:30 London  ",
        "Entry: 1 pt beyond ORB high/low; Target: 2R; Stop: opposite ORB side  ",
        "Cost: 2 pts round-trip spread (stop priority on same-bar ambiguity)  ",
        "",
    ]

    for symbol in SYMBOLS:
        for orb_win in ORB_WINDOWS_MIN:
            df = results[symbol][orb_win]
            lines += [f"### {symbol} -- {orb_win}-min ORB", ""]
            lines += [
                "| Direction | N days | Fill% | Win% | Avg win (pts) | Avg loss (pts) | Expectancy | Sharpe |",
                "|-----------|--------|-------|------|---------------|----------------|------------|--------|",
            ]
            for direction in ["long", "short"]:
                sub = df[df["direction"] == direction]
                row = _overall_row(direction, sub)
                if row:
                    lines.append(row)
            lines.append("")

            for direction in ["long", "short"]:
                sub = df[df["direction"] == direction]
                lines.append(f"**{direction.capitalize()} -- by weekday**  ")
                lines += _seg_table(sub, "weekday", WEEKDAY_ORDER)
                lines.append(f"**{direction.capitalize()} -- by gap open**  ")
                lines += _seg_table(sub, "gap_cat", GAP_LABELS)
                lines.append(f"**{direction.capitalize()} -- by ATR regime**  ")
                lines += _seg_table(sub, "atr_cat", ATR_LABELS)

    # Summary table
    lines += [
        "---",
        "",
        "### ORB EU Summary",
        "",
        "| Symbol | ORB window | Direction | N filled | Fill% | Expectancy (pts) | Sharpe | Go/No-go |",
        "|--------|------------|-----------|----------|-------|-----------------|--------|----------|",
    ]
    for symbol in SYMBOLS:
        for orb_win in ORB_WINDOWS_MIN:
            df = results[symbol][orb_win]
            for direction in ["long", "short"]:
                sub = df[df["direction"] == direction]
                m = _metrics_orb(sub)
                if not m:
                    continue
                verdict = "Go" if m["expectancy_pts"] > 0 else "No-go"
                lines.append(
                    f"| {symbol} | {orb_win}m | {direction} | {m['n_filled']} "
                    f"| {_fmt_plain(m['fill_rate_pct'])}% | {_fmt(m['expectancy_pts'])} "
                    f"| {_fmt(m['sharpe'], 3)} | {verdict} |"
                )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    results: dict[str, dict[int, pd.DataFrame]] = {}

    for symbol in SYMBOLS:
        print(f"\n{symbol}")
        df5    = _load_5min(symbol)
        print(f"  {len(df5):,} 5-min bars")
        daily  = _load_daily(symbol)
        atr_s  = _daily_atr(daily)
        s_open = _session_opens(df5, symbol)
        gap_s  = _gap_series(daily["close"], s_open)

        results[symbol] = {}
        for orb_win in ORB_WINDOWS_MIN:
            print(f"  {orb_win}-min ORB...", end=" ", flush=True)
            df = run_orb(symbol, df5, atr_s, gap_s, orb_win)
            n_days = len(df) // 2
            filled = df["filled"].sum()
            print(f"{n_days} days, {filled} filled ({filled / len(df) * 100:.1f}%)")
            results[symbol][orb_win] = df

    section = build_section(results)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"\nAppended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
