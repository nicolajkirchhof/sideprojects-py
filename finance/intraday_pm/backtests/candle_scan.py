"""
candle_scan.py
==============
Systematic search for the optimal OCO signal candle.

The Hougaard strategies fix bar index and timeframe by convention. This scan
tests every (timeframe x bar_index) combination in the first 2 hours of each
session, for both stop methods, to find whether the conventional choices are
truly optimal.

Search space
------------
Instruments:   IBDE40, IBGB100, IBUS500, IBUST100,
               IBEU50, IBFR40, IBES35, IBCH20, IBNL25,
               IBUS30, IBAU200, IBJP225, USGOLD
Timeframes:    5min, 10min, 15min, 30min
Bar indices:   first 2 hours of session (0-indexed from session open)
Stop methods:  atr  -- 20% of 14-day daily ATR (SRS style)
               bar_range -- bar range + 2x offset (ASRS style)
Exit:          2-bar trailing stop (fixed)
Entry offset:  2 pts (same as all Hougaard strategies)
Cost:          2 pts round-trip spread

Hougaard baselines embedded for comparison:
  IBDE40   SRS  15min bar 1 atr        EV +1.03  Sharpe +0.018
  IBDE40   ASRS  5min bar 3 bar_range  EV -0.27  Sharpe -0.007
  IBUST100 SRS  15min bar 1 atr        EV +3.66  Sharpe +0.052
  IBUST100 ASRS  5min bar 3 bar_range  EV +2.61  Sharpe +0.049
  IBGB100  SRS  15min bar 1 atr        EV -0.53  Sharpe -0.024
  IBUS500  SRS  15min bar 1 atr        EV -1.34  Sharpe -0.086
  IBUS500  ASRS  5min bar 3 bar_range  EV -1.39  Sharpe -0.116

Run from repo root:
    python finance/intraday_pm/backtests/candle_scan.py
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
    ATR_TRAIL_FACTOR,
    ATR_PERIOD,
    ENTRY_OFFSET_PTS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOLS = [
    # European index CFDs — CET/CEST session
    "IBDE40", "IBGB100", "IBEU50", "IBFR40", "IBES35", "IBCH20", "IBNL25",
    # US index CFDs — ET session
    "IBUS500", "IBUST100", "IBUS30",
    # Asian index CFDs
    "IBAU200", "IBJP225",
    # Gold CFD — London session as anchor
    "USGOLD",
]
SCHEMA    = "cfd"
START     = datetime(2020, 1, 1, tzinfo=timezone.utc)
END       = datetime(2026, 4, 1, tzinfo=timezone.utc)

TIMEFRAMES    = ["5min", "10min", "15min", "30min"]
STOP_METHODS  = ["atr", "bar_range"]
MAX_SCAN_HOURS = 2   # scan bar indices within first 2 hours of session

TZ_MAP = {
    # European — all CET/CEST (Europe/Berlin)
    "IBDE40":  ZoneInfo("Europe/Berlin"),
    "IBGB100": ZoneInfo("Europe/London"),
    "IBEU50":  ZoneInfo("Europe/Berlin"),
    "IBFR40":  ZoneInfo("Europe/Berlin"),   # Paris = CET
    "IBES35":  ZoneInfo("Europe/Berlin"),   # Madrid = CET
    "IBCH20":  ZoneInfo("Europe/Berlin"),   # Zurich = CET
    "IBNL25":  ZoneInfo("Europe/Berlin"),   # Amsterdam = CET
    # US
    "IBUS500":  ZoneInfo("America/New_York"),
    "IBUST100": ZoneInfo("America/New_York"),
    "IBUS30":   ZoneInfo("America/New_York"),
    # Asian
    "IBAU200":  ZoneInfo("Australia/Sydney"),
    "IBJP225":  ZoneInfo("Asia/Tokyo"),
    # Gold — London session
    "USGOLD":   ZoneInfo("Europe/London"),
}
SESSION_OPEN = {
    "IBDE40":  (9,  0),
    "IBGB100": (8,  0),
    "IBEU50":  (9,  0),
    "IBFR40":  (9,  0),
    "IBES35":  (9,  0),
    "IBCH20":  (9,  0),
    "IBNL25":  (9,  0),
    "IBUS500":  (9, 30),
    "IBUST100": (9, 30),
    "IBUS30":   (9, 30),
    "IBAU200":  (10,  0),
    "IBJP225":  (9,  0),
    "USGOLD":   (8,  0),
}
SESSION_CLOSE = {
    "IBDE40":  (17, 30),
    "IBGB100": (16, 30),
    "IBEU50":  (17, 30),
    "IBFR40":  (17, 30),
    "IBES35":  (17, 30),
    "IBCH20":  (17, 30),
    "IBNL25":  (17, 30),
    "IBUS500":  (16,  0),
    "IBUST100": (16,  0),
    "IBUS30":   (16,  0),
    "IBAU200":  (16,  0),
    "IBJP225":  (15, 30),
    "USGOLD":   (17,  0),
}

TF_MINUTES = {"5min": 5, "10min": 10, "15min": 15, "30min": 30}

# Known baselines: key = (symbol, tf, bar_idx, stop_method)
BASELINES: dict[tuple, tuple[str, float, float]] = {
    ("IBDE40",   "15min", 1, "atr"):        ("SRS DAX",       +1.03, +0.018),
    ("IBDE40",   "5min",  3, "bar_range"):  ("ASRS DAX",      -0.27, -0.007),
    ("IBUST100", "15min", 1, "atr"):        ("SRS NQ",        +3.66, +0.052),
    ("IBUST100", "5min",  3, "bar_range"):  ("ASRS NQ",       +2.61, +0.049),
    ("IBGB100",  "15min", 1, "atr"):        ("SRS GBX",       -0.53, -0.024),
    ("IBUS500",  "15min", 1, "atr"):        ("SRS ES",        -1.34, -0.086),
    ("IBUS500",  "5min",  3, "bar_range"):  ("ASRS ES",       -1.39, -0.116),
}

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"

TOP_N = 10  # top combinations to show per symbol in RESULTS.md


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tf_minutes(tf: str) -> int:
    return TF_MINUTES[tf]


def _max_bar_idx(tf: str) -> int:
    """Number of bars in MAX_SCAN_HOURS hours (exclusive upper bound)."""
    return (MAX_SCAN_HOURS * 60) // _tf_minutes(tf)


def _bar_time_str(symbol: str, bar_idx: int, tf: str) -> str:
    """Expected local time of bar at bar_idx from session open."""
    open_h, open_m = SESSION_OPEN[symbol]
    total_min = open_h * 60 + open_m + bar_idx * _tf_minutes(tf)
    return f"{total_min // 60:02d}:{total_min % 60:02d}"


def _is_baseline(symbol: str, tf: str, bar_idx: int, stop: str) -> str:
    """Return baseline label if this combination matches a known strategy, else ''."""
    entry = BASELINES.get((symbol, tf, bar_idx, stop))
    return entry[0] if entry else ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_bars(symbol: str, tf: str) -> pd.DataFrame:
    tz = TZ_MAP[symbol]
    df = get_bars(symbol, SCHEMA, START, END, period=tf)
    if df.empty:
        raise RuntimeError(f"No {tf} data for {symbol}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(tz)
    return df[["open", "high", "low", "close"]]


def _load_daily_atr(symbol: str) -> pd.Series:
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
# Session extraction
# ---------------------------------------------------------------------------
def _session_bars(day_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    open_h,  open_m  = SESSION_OPEN[symbol]
    close_h, close_m = SESSION_CLOSE[symbol]
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
# Core scan for one (symbol, tf, bar_idx, stop_method)
# ---------------------------------------------------------------------------
def _scan_one(
    sessions: dict[date, pd.DataFrame],
    atr_series: pd.Series,
    bar_idx: int,
    stop_method: str,
) -> dict:
    """
    Run OCO simulation for every session using bar_idx as signal bar.
    Returns aggregated metrics dict or empty dict if too few fills.
    """
    records: list[dict] = []

    for trade_date, session_bars in sessions.items():
        if len(session_bars) <= bar_idx:
            continue

        signal_bar = session_bars.iloc[bar_idx]
        remaining  = session_bars[session_bars.index > signal_bar.name]
        if remaining.empty:
            continue

        atr_val = atr_series.get(trade_date, np.nan)

        if stop_method == "atr":
            stop_pts = (
                atr_val * ATR_TRAIL_FACTOR if not np.isnan(atr_val)
                else signal_bar["high"] - signal_bar["low"] + 2 * ENTRY_OFFSET_PTS
            )
        else:  # bar_range
            stop_pts = signal_bar["high"] - signal_bar["low"] + 2 * ENTRY_OFFSET_PTS

        rec = _simulate_oco(signal_bar, remaining, stop_pts, use_2bar_trail=True)
        rec["date"] = trade_date
        records.append(rec)

    if not records:
        return {}
    return _metrics(pd.DataFrame(records))


# ---------------------------------------------------------------------------
# Run full scan for one symbol
# ---------------------------------------------------------------------------
def run_symbol_scan(symbol: str) -> pd.DataFrame:
    """
    Run the full candle scan for one symbol.
    Returns a DataFrame with one row per (tf, bar_idx, stop_method).
    """
    print(f"\n  {symbol}: loading ATR...")
    atr = _load_daily_atr(symbol)

    rows: list[dict] = []

    for tf in TIMEFRAMES:
        print(f"  {symbol}: {tf}...", end=" ", flush=True)
        try:
            bars = _load_bars(symbol, tf)
        except RuntimeError as e:
            print(f"skip ({e})")
            continue

        bars = bars.copy()
        bars["date"] = bars.index.date

        # Build sessions dict once per (symbol, tf)
        sessions: dict[date, pd.DataFrame] = {}
        for trade_date, day_df in bars.groupby("date"):
            sess = _session_bars(day_df, symbol)
            if len(sess) >= 2:
                sessions[trade_date] = sess

        max_idx = _max_bar_idx(tf)
        n_combos = max_idx * len(STOP_METHODS)
        done = 0

        for bar_idx in range(max_idx):
            for stop_method in STOP_METHODS:
                m = _scan_one(sessions, atr, bar_idx, stop_method)
                done += 1
                if m and m.get("n_filled", 0) >= 10:
                    baseline_label = _is_baseline(symbol, tf, bar_idx, stop_method)
                    rows.append({
                        "symbol":      symbol,
                        "tf":          tf,
                        "bar_idx":     bar_idx,
                        "bar_time":    _bar_time_str(symbol, bar_idx, tf),
                        "stop_method": stop_method,
                        "n_signals":   m["n_signals"],
                        "n_filled":    m["n_filled"],
                        "fill_pct":    m["fill_rate_pct"],
                        "win_pct":     m["win_rate_pct"],
                        "avg_win":     m["avg_win_pts"],
                        "avg_loss":    m["avg_loss_pts"],
                        "ev":          m["expectancy_pts"],
                        "sharpe":      m["sharpe"],
                        "baseline":    baseline_label,
                    })

        print(f"{len(sessions)} sessions, {n_combos} combinations")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------
def _row_line(r: pd.Series, mark_baseline: bool = True) -> str:
    tag = f" `{r['baseline']}`" if mark_baseline and r["baseline"] else ""
    return (
        f"| {r['tf']} | {r['bar_idx']} | {r['bar_time']} | {r['stop_method']} "
        f"| {r['n_filled']} | {_fmt_plain(r['fill_pct'])}% "
        f"| {_fmt_plain(r['win_pct'])}% | {_fmt(r['avg_win'])} "
        f"| {_fmt(r['avg_loss'])} | {_fmt(r['ev'])} | {_fmt(r['sharpe'], 3)}{tag} |"
    )


def _symbol_section(df: pd.DataFrame, symbol: str) -> list[str]:
    sub = df[df["symbol"] == symbol].copy()
    if sub.empty:
        return []

    sub_sorted = sub.sort_values("sharpe", ascending=False)

    lines: list[str] = [
        f"### {symbol}",
        "",
        f"Top {TOP_N} by Sharpe (all timeframes, both stop methods)  ",
        "",
        "| TF | Bar# | Time | Stop | N filled | Fill% | Win% | Avg win | Avg loss | EV (pts) | Sharpe |",
        "|-----|------|------|------|----------|-------|------|---------|----------|----------|--------|",
    ]

    # Top N non-baseline rows
    shown = 0
    baseline_rows = []
    for _, r in sub_sorted.iterrows():
        if r["baseline"]:
            baseline_rows.append(r)
        elif shown < TOP_N:
            lines.append(_row_line(r))
            shown += 1

    lines.append("")

    # Always show baselines separately so they're never lost
    if baseline_rows:
        lines += ["**Hougaard baselines for this symbol:**  ", ""]
        for r in baseline_rows:
            lines.append(_row_line(r))
        lines.append("")

    return lines


def build_section(all_results: pd.DataFrame) -> str:
    lines: list[str] = [
        "",
        "---",
        "",
        "## Candle Scan -- Optimal OCO Signal Bar",
        "",
        f"Generated: {date.today().isoformat()}  ",
        f"Period: {START.date().isoformat()} to {END.date().isoformat()}  ",
        "Search space: 5min/10min/15min/30min x first-2h bar indices x atr/bar_range stop  ",
        "Entry: OCO bracket +/-2 pts  |  Exit: 2-bar trailing stop  |  Cost: 2 pts  ",
        "",
    ]

    # Per-symbol sections
    for symbol in SYMBOLS:
        lines += _symbol_section(all_results, symbol)
        lines.append("---")
        lines.append("")

    # Cross-instrument top 20 by Sharpe
    lines += [
        "### Cross-Instrument Top 20 by Sharpe",
        "",
        "| Symbol | TF | Bar# | Time | Stop | N filled | Win% | EV (pts) | Sharpe | Baseline |",
        "|--------|-----|------|------|------|----------|------|----------|--------|----------|",
    ]
    top20 = all_results.nlargest(20, "sharpe")
    for _, r in top20.iterrows():
        tag = r["baseline"] if r["baseline"] else ""
        lines.append(
            f"| {r['symbol']} | {r['tf']} | {r['bar_idx']} | {r['bar_time']} "
            f"| {r['stop_method']} | {r['n_filled']} "
            f"| {_fmt_plain(r['win_pct'])}% | {_fmt(r['ev'])} | {_fmt(r['sharpe'], 3)} | {tag} |"
        )
    lines.append("")

    # Verdict: does scanning improve on existing Hougaard choices?
    lines += [
        "### Scan vs Hougaard Baselines",
        "",
        "| Symbol | Best scan: tf/bar/stop | Best EV | Best Sharpe | Baseline EV | Baseline Sharpe | Improvement |",
        "|--------|------------------------|---------|-------------|-------------|-----------------|-------------|",
    ]
    for symbol in SYMBOLS:
        sub = all_results[all_results["symbol"] == symbol]
        if sub.empty:
            continue
        best = sub.loc[sub["sharpe"].idxmax()]
        baseline_rows = sub[sub["baseline"] != ""]
        if baseline_rows.empty:
            bl_ev, bl_sh = np.nan, np.nan
            bl_label = "—"
        else:
            # Best baseline by Sharpe
            best_bl = baseline_rows.loc[baseline_rows["sharpe"].idxmax()]
            bl_ev   = best_bl["ev"]
            bl_sh   = best_bl["sharpe"]
            bl_label = best_bl["baseline"]
        delta_sharpe = best["sharpe"] - bl_sh if not np.isnan(bl_sh) else np.nan
        improvement = f"{delta_sharpe:+.3f}" if not np.isnan(delta_sharpe) else "—"
        lines.append(
            f"| {symbol} | {best['tf']}/{best['bar_idx']}/{best['stop_method']} "
            f"| {_fmt(best['ev'])} | {_fmt(best['sharpe'], 3)} "
            f"| {_fmt(bl_ev)} ({bl_label}) | {_fmt(bl_sh, 3)} | {improvement} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Candle scan: {len(SYMBOLS)} instruments x {len(TIMEFRAMES)} timeframes")
    print(f"Stop methods: {STOP_METHODS}  |  Scan window: first {MAX_SCAN_HOURS}h of session")

    all_results_list: list[pd.DataFrame] = []

    for symbol in SYMBOLS:
        df = run_symbol_scan(symbol)
        if not df.empty:
            all_results_list.append(df)

    if not all_results_list:
        print("No results collected.")
        return

    all_results = pd.concat(all_results_list, ignore_index=True)
    total_combos = len(all_results)
    go_combos    = (all_results["ev"] > 0).sum()
    print(f"\nTotal combinations: {total_combos}  |  Positive EV: {go_combos}")

    # Console preview: top 10 overall
    print("\nTop 10 by Sharpe (all instruments):")
    print(f"  {'Symbol':<10} {'TF':<6} {'Bar#':>4} {'Time':>6} {'Stop':<10} "
          f"{'N':>5} {'Win%':>6} {'EV':>7} {'Sharpe':>8} {'Baseline'}")
    for _, r in all_results.nlargest(10, "sharpe").iterrows():
        tag = f" <-- {r['baseline']}" if r["baseline"] else ""
        print(
            f"  {r['symbol']:<10} {r['tf']:<6} {r['bar_idx']:>4} {r['bar_time']:>6} "
            f"{r['stop_method']:<10} {r['n_filled']:>5} {r['win_pct']:>5.1f}% "
            f"{r['ev']:>+7.2f} {r['sharpe']:>+8.3f}{tag}"
        )

    section = build_section(all_results)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"\nAppended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
