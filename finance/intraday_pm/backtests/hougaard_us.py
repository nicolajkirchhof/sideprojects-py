"""
hougaard_us.py
==============
Hougaard bracket strategies on US instruments (IBUS500 + IBUST100), ET session.

BT-4-S5 -- ASRS (After Session Range Setup):
  Signal bar: 4th 5-min bar of the session (09:45-09:50 ET).
  Fallback: if bar range < 5 pts, use 5th bar.
  Entry: OCO bracket -- buy-stop 2 pts above bar high, sell-stop 2 pts below bar low.
  Stop: symmetric (bar range + 4 pts total risk).
  Exit: 2-bar trailing stop on 5-min bars; EOD fallback (16:00 ET).

BT-4-S6 -- SRS (Session Range Setup):
  Signal bar: 2nd 15-min bar (09:45-10:00 ET).
  Entry: same OCO bracket +/- 2 pts.
  Stop: 20% of prior 14-day daily ATR.
  Exit: 2-bar trailing stop on 15-min bars; EOD fallback.

Both strategies cost 2 pts round-trip spread.

Run from repo root:
    python finance/intraday_pm/backtests/hougaard_us.py
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
from finance.intraday_pm.backtests.hougaard_dax import (
    _simulate_oco,
    _metrics,
    _fmt,
    _fmt_plain,
    _overall_table,
    _segmented_table,
    ATR_TRAIL_FACTOR,
    ATR_PERIOD,
    ENTRY_OFFSET_PTS,
    MIN_BAR_RANGE_PTS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOLS  = ["IBUS500", "IBUST100"]
SCHEMA   = "cfd"
ET_TZ    = ZoneInfo("America/New_York")
START    = datetime(2020, 1, 1, tzinfo=timezone.utc)
END      = datetime(2026, 4, 1, tzinfo=timezone.utc)

SESSION_OPEN  = (9,  30)  # 09:30 ET
SESSION_CLOSE = (16,  0)  # 16:00 ET

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
RANGE_LABELS  = ["narrow (<10 pts)", "normal (10-25 pts)", "wide (>25 pts)"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_bars(symbol: str, period: str) -> pd.DataFrame:
    df = get_bars(symbol, SCHEMA, START, END, period=period)
    if df.empty:
        raise RuntimeError(f"No {period} data for {symbol}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(ET_TZ)
    return df[["open", "high", "low", "close"]]


def load_daily_atr(symbol: str) -> pd.Series:
    df = get_bars(symbol, SCHEMA, START, END, period="1D")
    df = df.rename(columns={"h": "high", "l": "low", "c": "close"})
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()
    atr.index = atr.index.date
    return atr


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _session_bars(day_df: pd.DataFrame) -> pd.DataFrame:
    """Return bars within the ET session (09:30-16:00)."""
    open_h,  open_m  = SESSION_OPEN
    close_h, close_m = SESSION_CLOSE
    mask = (
        (
            (day_df.index.hour == open_h) & (day_df.index.minute >= open_m)
        ) | (day_df.index.hour > open_h)
    ) & (
        (day_df.index.hour < close_h) |
        ((day_df.index.hour == close_h) & (day_df.index.minute <= close_m))
    )
    return day_df[mask]


# ---------------------------------------------------------------------------
# BT-4-S5: ASRS -- 4th 5-min bar at 09:45 ET
# ---------------------------------------------------------------------------
def run_asrs(symbol: str, df5: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Simulate ASRS: OCO bracket on the 4th (or 5th fallback) 5-min bar."""
    records: list[dict] = []
    df5 = df5.copy()
    df5["date"] = df5.index.date

    for trade_date, day_df in df5.groupby("date"):
        session = _session_bars(day_df)
        # Open-hour bars only (09:30-09:59) for signal selection
        open_bars = session[
            (session.index.hour == SESSION_OPEN[0]) &
            (session.index.minute >= SESSION_OPEN[1])
        ]
        if len(open_bars) < 4:
            continue

        signal_bar = open_bars.iloc[3]  # 4th bar = 09:45 ET
        if (signal_bar["high"] - signal_bar["low"]) < MIN_BAR_RANGE_PTS:
            if len(open_bars) >= 5:
                signal_bar = open_bars.iloc[4]
            else:
                continue

        remaining = session[session.index > signal_bar.name]
        stop_pts  = signal_bar["high"] - signal_bar["low"] + 2 * ENTRY_OFFSET_PTS
        atr_val   = atr_series.get(trade_date, np.nan)

        rec = _simulate_oco(signal_bar, remaining, stop_pts, use_2bar_trail=True)
        rec["date"]    = trade_date
        rec["weekday"] = signal_bar.name.strftime("%A")
        rec["atr"]     = atr_val
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# BT-4-S6: SRS -- 2nd 15-min bar at 09:45 ET
# ---------------------------------------------------------------------------
def run_srs(symbol: str, df15: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Simulate SRS: OCO bracket on the 2nd 15-min bar (09:45-10:00 ET)."""
    records: list[dict] = []
    df15 = df15.copy()
    df15["date"] = df15.index.date

    for trade_date, day_df in df15.groupby("date"):
        open_bars = day_df[
            (day_df.index.hour == SESSION_OPEN[0]) &
            (day_df.index.minute >= SESSION_OPEN[1])
        ]
        if len(open_bars) < 2:
            continue

        signal_bar = open_bars.iloc[1]  # 2nd 15-min bar = 09:45 ET

        eod_mask = (day_df.index > signal_bar.name) & (
            (day_df.index.hour < SESSION_CLOSE[0]) |
            ((day_df.index.hour == SESSION_CLOSE[0]) &
             (day_df.index.minute <= SESSION_CLOSE[1]))
        )
        remaining = day_df[eod_mask]

        atr_val  = atr_series.get(trade_date, np.nan)
        stop_pts = (
            atr_val * ATR_TRAIL_FACTOR if not np.isnan(atr_val)
            else signal_bar["high"] - signal_bar["low"] + 2 * ENTRY_OFFSET_PTS
        )

        rec = _simulate_oco(signal_bar, remaining, stop_pts, use_2bar_trail=True)
        rec["date"]    = trade_date
        rec["weekday"] = signal_bar.name.strftime("%A")
        rec["atr"]     = atr_val
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------
def build_section(
    results: dict[str, dict[str, pd.DataFrame]],
) -> str:
    range_bins = [0, 10, 25, float("inf")]

    lines: list[str] = [
        "",
        "---",
        "",
        "## Hougaard US Bracket Strategies -- IBUS500 + IBUST100",
        "",
        f"Generated: {date.today().isoformat()}  ",
        "Instruments: IBUS500, IBUST100, ET time  ",
        f"Period: {START.date().isoformat()} to {END.date().isoformat()}  ",
        "Entry: OCO bracket +/-2 pts of signal bar high/low  ",
        "Exit: 2-bar trailing stop or EOD (16:00 ET)  ",
        "Cost: 2 pts round-trip spread  ",
        "",
    ]

    for symbol in SYMBOLS:
        lines += [f"---", "", f"### {symbol}", ""]

        for strategy, label, sig_desc in [
            ("asrs", "BT-4-S5 ASRS", "4th 5-min bar at 09:45 ET"),
            ("srs",  "BT-4-S6 SRS",  "2nd 15-min bar at 09:45 ET"),
        ]:
            df = results[symbol][strategy].copy()
            df["range_cat"] = pd.cut(
                df["bar_range"], bins=range_bins, labels=RANGE_LABELS, right=False
            )
            df["weekday"] = pd.Categorical(
                df["weekday"], categories=WEEKDAY_ORDER, ordered=True
            )

            lines += [f"#### {label} ({sig_desc})", ""]
            lines += _overall_table(df, f"{label} {symbol}")
            lines += _segmented_table(df, "range_cat", RANGE_LABELS, f"{label} {symbol}")
            lines += _segmented_table(df, "weekday", WEEKDAY_ORDER, f"{label} {symbol}")

    # Summary
    lines += [
        "---",
        "",
        "### Hougaard US Summary",
        "",
        "| Symbol | Strategy | N filled | Win% | Expectancy (pts) | Sharpe | Go/No-go |",
        "|--------|----------|----------|------|-----------------|--------|----------|",
    ]
    for symbol in SYMBOLS:
        for strategy, label in [("asrs", "ASRS"), ("srs", "SRS")]:
            df = results[symbol][strategy]
            m = _metrics(df)
            if not m:
                continue
            verdict = "Go" if m["expectancy_pts"] > 0 else "No-go"
            lines.append(
                f"| {symbol} | {label} | {m['n_filled']} "
                f"| {_fmt_plain(m['win_rate_pct'])}% | {_fmt(m['expectancy_pts'])} "
                f"| {_fmt(m['sharpe'], 3)} | {verdict} |"
            )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    results: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in SYMBOLS:
        print(f"\n{symbol}")
        df5  = load_bars(symbol, "5min")
        df15 = load_bars(symbol, "15min")
        print(f"  5-min: {len(df5):,} bars  |  15-min: {len(df15):,} bars")

        atr = load_daily_atr(symbol)

        print("  Running ASRS (BT-4-S5)...", end=" ", flush=True)
        asrs = run_asrs(symbol, df5, atr)
        filled = asrs["filled_direction"].notna().sum()
        print(f"{len(asrs)} signal days, {filled} filled ({filled / len(asrs) * 100:.1f}%)")

        print("  Running SRS (BT-4-S6)...", end=" ", flush=True)
        srs = run_srs(symbol, df15, atr)
        filled = srs["filled_direction"].notna().sum()
        print(f"{len(srs)} signal days, {filled} filled ({filled / len(srs) * 100:.1f}%)")

        results[symbol] = {"asrs": asrs, "srs": srs}

    section = build_section(results)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"\nAppended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
