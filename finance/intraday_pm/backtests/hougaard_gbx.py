"""
hougaard_gbx.py
===============
BT-4-S4: Hougaard SRS on IBGB100 (FTSE 100 CFD), London session.

Strategy mirrors BT-4-S3 (hougaard_dax.py SRS) with IBGB100-specific parameters:
  Instrument:  IBGB100
  Session:     08:00-16:30 London (Europe/London)
  Signal bar:  2nd 15-min bar (08:15-08:30 London)
  Entry:       OCO bracket -- buy-stop 2 pts above bar high, sell-stop 2 pts below bar low.
  Stop:        20% of prior 14-day daily ATR (scaled to FTSE level).
  Exit:        2-bar trailing stop on 15-min bars; EOD fallback (16:30 London).
  Cost:        2 pts round-trip spread.

Run from repo root:
    python finance/intraday_pm/backtests/hougaard_gbx.py
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
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL     = "IBGB100"
SCHEMA     = "cfd"
LONDON_TZ  = ZoneInfo("Europe/London")
START      = datetime(2020, 1, 1, tzinfo=timezone.utc)
END        = datetime(2026, 4, 1, tzinfo=timezone.utc)

SESSION_OPEN  = (8,  0)   # 08:00 London
SESSION_CLOSE = (16, 30)  # 16:30 London

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
RANGE_LABELS  = ["narrow (<10 pts)", "normal (10-25 pts)", "wide (>25 pts)"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_bars(period: str) -> pd.DataFrame:
    df = get_bars(SYMBOL, SCHEMA, START, END, period=period)
    if df.empty:
        raise RuntimeError(f"No {period} data for {SYMBOL}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(LONDON_TZ)
    return df[["open", "high", "low", "close"]]


def load_daily_atr() -> pd.Series:
    df = get_bars(SYMBOL, SCHEMA, START, END, period="1D")
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
# BT-4-S4: SRS on IBGB100 -- 2nd 15-min bar at 08:15 London
# ---------------------------------------------------------------------------
def run_srs_gbx(df15: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """
    Simulate SRS on IBGB100: OCO bracket on the 2nd 15-min bar (08:15-08:30 London).

    Stop is 20% of the prior 14-day daily ATR, falling back to bar range + 2x offset
    if ATR is unavailable.
    """
    records: list[dict] = []
    df15 = df15.copy()
    df15["date"] = df15.index.date

    for trade_date, day_df in df15.groupby("date"):
        open_bars = day_df[day_df.index.hour == SESSION_OPEN[0]]
        if len(open_bars) < 2:
            continue

        signal_bar = open_bars.iloc[1]  # 2nd 15-min bar = 08:15 London

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
def build_section(srs: pd.DataFrame) -> str:
    range_bins = [0, 10, 25, float("inf")]
    srs = srs.copy()
    srs["range_cat"] = pd.cut(
        srs["bar_range"], bins=range_bins, labels=RANGE_LABELS, right=False
    )
    srs["weekday"] = pd.Categorical(srs["weekday"], categories=WEEKDAY_ORDER, ordered=True)

    lines: list[str] = [
        "",
        "---",
        "",
        "## Hougaard SRS -- IBGB100 (BT-4-S4)",
        "",
        f"Generated: {date.today().isoformat()}  ",
        "Instrument: IBGB100, London time  ",
        f"Period: {START.date().isoformat()} to {END.date().isoformat()}  ",
        "Signal: 2nd 15-min bar at 08:15 London  ",
        "Entry: OCO bracket +/-2 pts of signal bar high/low  ",
        "Stop: 20% of 14-day daily ATR  ",
        "Exit: 2-bar trailing stop or EOD (16:30 London)  ",
        "Cost: 2 pts round-trip spread  ",
        "",
        "---",
        "",
        "### BT-4-S4: SRS (2nd 15-min bar at 08:15 London)",
        "",
    ]

    lines += _overall_table(srs, "SRS IBGB100")
    lines += _segmented_table(srs, "range_cat", RANGE_LABELS, "SRS IBGB100")
    lines += _segmented_table(srs, "weekday", WEEKDAY_ORDER, "SRS IBGB100")

    m = _metrics(srs)
    if m:
        verdict = "Go" if m["expectancy_pts"] > 0 else "No-go"
        lines += [
            "---",
            "",
            "### SRS IBGB100 Verdict",
            "",
            f"**{verdict}** -- N filled: {m['n_filled']} | Win%: {_fmt_plain(m['win_rate_pct'])}% "
            f"| Expectancy: {_fmt(m['expectancy_pts'])} pts | Sharpe: {_fmt(m['sharpe'], 3)}",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading bars for {SYMBOL}...")
    df15 = load_bars("15min")
    print(f"  15-min: {len(df15):,} bars")

    print("Loading ATR...")
    atr = load_daily_atr()

    print("Running SRS (BT-4-S4)...")
    srs = run_srs_gbx(df15, atr)
    filled = srs["filled_direction"].notna().sum()
    print(f"  {len(srs)} signal days, {filled} filled ({filled / len(srs) * 100:.1f}%)")

    section = build_section(srs)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"Appended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
