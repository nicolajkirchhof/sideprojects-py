"""
hougaard_fomc.py
================
BT-4-S4: Hougaard Rule of 4 — FOMC bracket strategy.

Signal: 4th 10-min bar after the FOMC announcement (14:00 ET).
Entry: OCO bracket ±2 pts of signal bar high/low.
Stop: opposite bracket side.
Exit: 2-bar trailing stop; 3-hour session cutoff (17:00 ET).
Cost: 2 pts round-trip spread.

Comparison: Rule of 4 vs random bracket on non-FOMC days at the same time (14:40 ET).

Run from repo root:
    python finance/intraday_pm/backtests/hougaard_fomc.py
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
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL = "IBUS500"
SCHEMA = "cfd"
ET_TZ = ZoneInfo("America/New_York")
START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 4, 1, tzinfo=timezone.utc)

ANNOUNCEMENT_HOUR_ET = 14
ANNOUNCEMENT_MINUTE_ET = 0
# 4th 10-min bar starts 30 min after announcement = 14:30 ET
SIGNAL_BAR_MINUTE_OFFSET = 30
SESSION_CUTOFF_HOUR_ET = 17
SPREAD_COST_PTS = 2.0

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"

# ---------------------------------------------------------------------------
# FOMC announcement dates (2020-2024, 8 meetings/year = 40 events).
# Sources: Federal Reserve published calendars.
# ---------------------------------------------------------------------------
FOMC_DATES = [
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]
FOMC_DATE_SET = set(pd.to_datetime(FOMC_DATES).date)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_10min() -> pd.DataFrame:
    df = get_bars(SYMBOL, SCHEMA, START, END, period="10min")
    if df.empty:
        raise RuntimeError(f"No 10-min data for {SYMBOL}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(ET_TZ)
    return df[["open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _run_bracket_at(trade_date: date, signal_bar: pd.Series, day_df: pd.DataFrame) -> dict:
    bar_range = signal_bar["high"] - signal_bar["low"]
    stop_pts = bar_range + 2 * SPREAD_COST_PTS

    cutoff = signal_bar.name.replace(hour=SESSION_CUTOFF_HOUR_ET, minute=0)
    remaining = day_df[(day_df.index > signal_bar.name) & (day_df.index <= cutoff)]

    rec = _simulate_oco(signal_bar, remaining, stop_pts, use_2bar_trail=True)
    rec["date"] = trade_date
    return rec


def run_fomc(df10: pd.DataFrame) -> pd.DataFrame:
    """Simulate Rule of 4 on FOMC days: 4th 10-min bar after 14:00 ET."""
    records: list[dict] = []
    df10["date"] = df10.index.date

    for trade_date, day_df in df10.groupby("date"):
        if trade_date not in FOMC_DATE_SET:
            continue

        # Signal bar: 4th 10-min bar after 14:00 = bars at 14:00, 14:10, 14:20, 14:30
        # The 4th bar is 14:30 ET (opens 14:30, closes 14:40)
        announcement_mask = (
            (day_df.index.hour == ANNOUNCEMENT_HOUR_ET) &
            (day_df.index.minute >= ANNOUNCEMENT_MINUTE_ET)
        ) | (day_df.index.hour > ANNOUNCEMENT_HOUR_ET)
        after_ann = day_df[announcement_mask]

        if len(after_ann) < 4:
            continue

        signal_bar = after_ann.iloc[3]  # 4th bar = index 3
        rec = _run_bracket_at(trade_date, signal_bar, day_df)
        rec["type"] = "fomc"
        records.append(rec)

    return pd.DataFrame(records)


def run_control(df10: pd.DataFrame) -> pd.DataFrame:
    """Simulate same bracket on non-FOMC days at 14:30 ET as control group."""
    records: list[dict] = []
    df10["date"] = df10.index.date

    for trade_date, day_df in df10.groupby("date"):
        if trade_date in FOMC_DATE_SET:
            continue
        # Signal bar: 10-min bar opening at 14:30 ET
        signal_mask = (day_df.index.hour == 14) & (day_df.index.minute == 30)
        if not any(signal_mask):
            continue
        signal_bar = day_df[signal_mask].iloc[0]
        rec = _run_bracket_at(trade_date, signal_bar, day_df)
        rec["type"] = "control"
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Markdown section
# ---------------------------------------------------------------------------
def build_section(fomc: pd.DataFrame, control: pd.DataFrame) -> str:
    lines: list[str] = [
        "",
        "---",
        "",
        "## Hougaard Rule of 4 — FOMC Bracket (IBUS500)",
        "",
        f"Generated: {date.today().isoformat()}  ",
        "Instrument: IBUS500, 10-min bars, ET timezone  ",
        f"Period: {START.date().isoformat()} -> {END.date().isoformat()}  ",
        f"FOMC events: {len(fomc)} days  ",
        "Signal: 4th 10-min bar after 14:00 ET announcement (bar opens 14:30 ET)  ",
        "Entry: OCO bracket ±2 pts; Stop: opposite bracket side  ",
        "Exit: 2-bar trailing stop or 17:00 ET cutoff  ",
        "Cost: 2 pts round-trip spread  ",
        "Control: same bracket on non-FOMC days at 14:30 ET  ",
        "",
        "| Group | N days | Fill% | Win% | Avg win (pts) | Avg loss (pts) | Expectancy | Sharpe |",
        "|-------|--------|-------|------|---------------|----------------|------------|--------|",
    ]

    for label, df in [("FOMC Rule of 4", fomc), ("Non-FOMC control", control)]:
        m = _metrics(df)
        if not m:
            continue
        lines.append(
            f"| {label} | {m['n_signals']} | {_fmt_plain(m['fill_rate_pct'])}% "
            f"| {_fmt_plain(m['win_rate_pct'])}% | {_fmt(m['avg_win_pts'])} "
            f"| {_fmt(m['avg_loss_pts'])} | {_fmt(m['expectancy_pts'])} | {_fmt(m['sharpe'], 3)} |"
        )

    fomc_m = _metrics(fomc)
    ctrl_m = _metrics(control)
    fomc_ev = fomc_m.get("expectancy_pts", np.nan) if fomc_m else np.nan
    ctrl_ev = ctrl_m.get("expectancy_pts", np.nan) if ctrl_m else np.nan
    edge = fomc_ev - ctrl_ev if not np.isnan(fomc_ev) and not np.isnan(ctrl_ev) else np.nan

    verdict = "Go" if (not np.isnan(fomc_ev) and fomc_ev > 0 and edge > 0) else "No-go"

    lines += [
        "",
        f"FOMC edge vs control: {_fmt(edge)} pts/trade  ",
        f"**Verdict: {verdict}**",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading 10-min bars...")
    df10 = load_10min()
    print(f"  {len(df10):,} bars")

    print(f"Running FOMC Rule of 4 ({len(FOMC_DATE_SET)} events)...")
    fomc = run_fomc(df10)
    print(f"  {len(fomc)} FOMC days, {fomc['filled_direction'].notna().sum()} filled")

    print("Running non-FOMC control...")
    control = run_control(df10)
    print(f"  {len(control)} control days, {control['filled_direction'].notna().sum()} filled")

    section = build_section(fomc, control)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"Appended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
