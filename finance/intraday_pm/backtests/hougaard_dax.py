"""
hougaard_dax.py
===============
Backtests two Hougaard bracket strategies on IBDE40 (DAX), both using an OCO
bracket placed around a specific signal bar at the Frankfurt session open.

BT-4-S2 — ASRS (After Session Range Setup):
  Signal bar: 4th 5-min bar of the session (09:15–09:20 Frankfurt time).
  Fallback: if bar range < 5 pts, use 5th bar.
  Entry: OCO bracket — buy-stop 2 pts above bar high, sell-stop 2 pts below bar low.
  Stop: opposite bracket side (symmetric; total risk = bar range + 4 pts).
  Exit: 2-bar trailing stop on 5-min bars; EOD fallback (17:30 Frankfurt).

BT-4-S3 — SRS (Session Range Setup):
  Signal bar: 2nd 15-min bar (09:15–09:30 Frankfurt time).
  Entry: same OCO bracket ± 2 pts.
  Stop: 20% of prior 14-day ATR (scaled to DAX level).
  Exit: 2-bar trailing stop on 15-min bars; EOD fallback.

Both strategies cost 2 pts round-trip spread.

Run from repo root:
    python finance/intraday_pm/backtests/hougaard_dax.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from finance.utils.intraday import get_bars

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL = "IBDE40"
SCHEMA = "cfd"
FRANKFURT_TZ = ZoneInfo("Europe/Berlin")
START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 1, tzinfo=timezone.utc)

SESSION_OPEN = (9, 0)    # 09:00 Frankfurt
SESSION_CLOSE = (17, 30) # 17:30 Frankfurt
ENTRY_OFFSET_PTS = 2.0
SPREAD_COST_PTS = 2.0
ATR_TRAIL_FACTOR = 0.20
ATR_PERIOD = 14
MIN_BAR_RANGE_PTS = 5.0   # BT-4-S2 fallback: if 4th bar range < this, use 5th

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
RANGE_LABELS = ["narrow (<10 pts)", "normal (10-25 pts)", "wide (>25 pts)"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_bars(period: str) -> pd.DataFrame:
    df = get_bars(SYMBOL, SCHEMA, START, END, period=period)
    if df.empty:
        raise RuntimeError(f"No {period} data for {SYMBOL}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(FRANKFURT_TZ)
    return df[["open", "high", "low", "close"]]


def load_daily_atr() -> pd.Series:
    df = get_bars(SYMBOL, SCHEMA, START, END, period="1D")
    df = df.rename(columns={"h": "high", "l": "low", "c": "close"})
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()
    atr.index = atr.index.date
    return atr


# ---------------------------------------------------------------------------
# OCO bracket simulation (shared)
# ---------------------------------------------------------------------------
def _simulate_oco(
    signal_bar: pd.Series,
    remaining_bars: pd.DataFrame,
    stop_pts: float,
    exit_mode: str = "2bar_trail",
    atr_pts: float = np.nan,
    entry_offset: float = ENTRY_OFFSET_PTS,
    spread_cost: float = SPREAD_COST_PTS,
) -> dict:
    """
    Simulate an OCO bracket trade from a signal bar.

    Both long and short sides are tracked simultaneously. The first side filled
    wins the trade; the other side is cancelled.

    exit_mode options:
        "2bar_trail" — trailing stop behind last-2-bar high/low
        "atr_trail"  — trailing stop at price ± ATR * ATR_TRAIL_FACTOR
        "fixed_2r"   — hard stop at opposite bracket side, fixed 2R take-profit

    entry_offset and spread_cost default to module-level constants so existing
    callers are unaffected; pass explicit values for non-equity instruments.

    Returns a dict with keys: filled_direction, result_pts, entry, win, bar_range.
    """
    bar_range = signal_bar["high"] - signal_bar["low"]
    long_entry = signal_bar["high"] + entry_offset
    short_entry = signal_bar["low"] - entry_offset
    long_stop = long_entry - stop_pts
    short_stop = short_entry + stop_pts

    base = {
        "filled_direction": None,
        "result_pts": np.nan,
        "entry": np.nan,
        "win": np.nan,
        "bar_range": bar_range,
    }

    if remaining_bars.empty:
        return base

    # Find which side is filled first
    filled_dir = None
    filled_entry = np.nan
    filled_idx = None

    for idx, bar in remaining_bars.iterrows():
        hit_long = bar["high"] >= long_entry
        hit_short = bar["low"] <= short_entry

        if hit_long and hit_short:
            # Same bar triggers both — assume the first touch wins; use open direction
            if bar["open"] >= signal_bar["high"]:
                filled_dir = "long"
                filled_entry = long_entry
            else:
                filled_dir = "short"
                filled_entry = short_entry
            filled_idx = idx
            break
        elif hit_long:
            filled_dir = "long"
            filled_entry = long_entry
            filled_idx = idx
            break
        elif hit_short:
            filled_dir = "short"
            filled_entry = short_entry
            filled_idx = idx
            break

    if filled_dir is None:
        return base

    base["filled_direction"] = filled_dir
    base["entry"] = filled_entry
    stop = long_stop if filled_dir == "long" else short_stop
    post_entry = remaining_bars.loc[remaining_bars.index > filled_idx]

    # Exit dispatch
    if exit_mode == "2bar_trail":
        stop_val, exit_price = _2bar_trail(
            filled_dir, stop, filled_entry, post_entry
        )
    elif exit_mode == "atr_trail":
        stop_val, exit_price = _atr_trail(
            filled_dir, stop, filled_entry, atr_pts, post_entry
        )
    elif exit_mode == "fixed_2r":
        stop_val, exit_price = _fixed_2r(
            filled_dir, stop, filled_entry, post_entry
        )
    else:
        raise ValueError(f"Unknown exit_mode: {exit_mode!r}")

    if filled_dir == "long":
        result = (exit_price - filled_entry) - spread_cost
    else:
        result = (filled_entry - exit_price) - spread_cost

    base["result_pts"] = result
    base["win"] = bool(result > 0)
    return base


def _2bar_trail(
    direction: str, initial_stop: float, entry: float, bars: pd.DataFrame
) -> tuple[float, float]:
    stop = initial_stop
    history: list[pd.Series] = []

    for _, bar in bars.iterrows():
        if direction == "long" and bar["low"] <= stop:
            return stop, stop
        if direction == "short" and bar["high"] >= stop:
            return stop, stop

        history.append(bar)
        if len(history) >= 2:
            if direction == "long":
                candidate = min(b["low"] for b in history[-2:])
                if candidate > stop:
                    stop = candidate
            else:
                candidate = max(b["high"] for b in history[-2:])
                if candidate < stop:
                    stop = candidate

    eod_exit = bars.iloc[-1]["close"] if not bars.empty else entry
    return stop, eod_exit


def _atr_trail(
    direction: str, initial_stop: float, entry: float,
    atr_pts: float, bars: pd.DataFrame
) -> tuple[float, float]:
    stop = initial_stop
    dist = atr_pts * ATR_TRAIL_FACTOR if not np.isnan(atr_pts) else abs(initial_stop - entry)

    for _, bar in bars.iterrows():
        if direction == "long" and bar["low"] <= stop:
            return stop, stop
        if direction == "short" and bar["high"] >= stop:
            return stop, stop

        if direction == "long":
            candidate = bar["close"] - dist
            if candidate > stop:
                stop = candidate
        else:
            candidate = bar["close"] + dist
            if candidate < stop:
                stop = candidate

    eod_exit = bars.iloc[-1]["close"] if not bars.empty else entry
    return stop, eod_exit


def _fixed_2r(
    direction: str, stop: float, entry: float, bars: pd.DataFrame
) -> tuple[float, float]:
    """
    Fixed 2R exit: hard stop at `stop`, take-profit at entry ± 2 * risk.

    Within each bar, stop is checked before target (conservative).
    Returns (stop, exit_price).
    """
    if bars.empty:
        return stop, entry

    risk = abs(entry - stop)
    target = entry + 2 * risk if direction == "long" else entry - 2 * risk

    for _, bar in bars.iterrows():
        if direction == "long":
            if bar["low"] <= stop:
                return stop, stop
            if bar["high"] >= target:
                return stop, target
        else:
            if bar["high"] >= stop:
                return stop, stop
            if bar["low"] <= target:
                return stop, target

    eod_exit = bars.iloc[-1]["close"]
    return stop, eod_exit


# ---------------------------------------------------------------------------
# BT-4-S2: ASRS — 4th 5-min bar
# ---------------------------------------------------------------------------
def run_asrs(df5: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Simulate ASRS: OCO bracket on the 4th (or 5th fallback) 5-min bar."""
    records: list[dict] = []
    df5["date"] = df5.index.date

    for trade_date, day_df in df5.groupby("date"):
        session_bars = day_df[
            (day_df.index.hour == SESSION_OPEN[0]) & (day_df.index.minute >= 0) |
            (day_df.index.hour > SESSION_OPEN[0])
        ]
        session_bars = session_bars[
            (session_bars.index.hour < SESSION_CLOSE[0]) |
            ((session_bars.index.hour == SESSION_CLOSE[0]) &
             (session_bars.index.minute <= SESSION_CLOSE[1]))
        ]
        open_bars = session_bars[
            (session_bars.index.hour == SESSION_OPEN[0])
        ]
        if len(open_bars) < 4:
            continue

        signal_bar = open_bars.iloc[3]  # 4th bar = index 3 (09:15)
        # Fallback: if range too narrow, use 5th bar
        if (signal_bar["high"] - signal_bar["low"]) < MIN_BAR_RANGE_PTS:
            if len(open_bars) >= 5:
                signal_bar = open_bars.iloc[4]
            else:
                continue

        eod_mask = session_bars.index > signal_bar.name
        remaining = session_bars[eod_mask]

        stop_pts = signal_bar["high"] - signal_bar["low"] + 2 * ENTRY_OFFSET_PTS
        atr_val = atr_series.get(trade_date, np.nan)

        rec = _simulate_oco(signal_bar, remaining, stop_pts, exit_mode="2bar_trail")
        rec["date"] = trade_date
        rec["weekday"] = signal_bar.name.strftime("%A")
        rec["atr"] = atr_val
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# BT-4-S3: SRS — 2nd 15-min bar
# ---------------------------------------------------------------------------
def run_srs(df15: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Simulate SRS: OCO bracket on the 2nd 15-min bar (09:15–09:30 Frankfurt)."""
    records: list[dict] = []
    df15["date"] = df15.index.date

    for trade_date, day_df in df15.groupby("date"):
        open_bars = day_df[day_df.index.hour == SESSION_OPEN[0]]
        if len(open_bars) < 2:
            continue

        signal_bar = open_bars.iloc[1]  # 2nd 15-min bar = 09:15–09:30
        eod_mask = (day_df.index > signal_bar.name) & (
            (day_df.index.hour < SESSION_CLOSE[0]) |
            ((day_df.index.hour == SESSION_CLOSE[0]) &
             (day_df.index.minute <= SESSION_CLOSE[1]))
        )
        remaining = day_df[eod_mask]

        atr_val = atr_series.get(trade_date, np.nan)
        # SRS stop = 20% of daily ATR
        stop_pts = atr_val * ATR_TRAIL_FACTOR if not np.isnan(atr_val) else (
            signal_bar["high"] - signal_bar["low"] + 2 * ENTRY_OFFSET_PTS
        )

        rec = _simulate_oco(signal_bar, remaining, stop_pts, exit_mode="2bar_trail")
        rec["date"] = trade_date
        rec["weekday"] = signal_bar.name.strftime("%A")
        rec["atr"] = atr_val
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics + segmentation
# ---------------------------------------------------------------------------
def _metrics(trades: pd.DataFrame, result_col: str = "result_pts") -> dict:
    completed = trades[trades[result_col].notna() & trades["filled_direction"].notna()]
    if completed.empty:
        return {}
    n = len(completed)
    wins = completed[completed[result_col] > 0]
    losses = completed[completed[result_col] <= 0]
    win_rate = len(wins) / n * 100
    avg_win = wins[result_col].mean() if not wins.empty else 0.0
    avg_loss = losses[result_col].mean() if not losses.empty else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    sharpe = completed[result_col].mean() / completed[result_col].std() if completed[result_col].std() > 0 else 0.0
    fill_rate = n / len(trades) * 100
    return {
        "n_signals": len(trades),
        "n_filled": n,
        "fill_rate_pct": fill_rate,
        "win_rate_pct": win_rate,
        "avg_win_pts": avg_win,
        "avg_loss_pts": avg_loss,
        "expectancy_pts": expectancy,
        "total_pts": completed[result_col].sum(),
        "sharpe": sharpe,
    }


def _fmt(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:+.{decimals}f}"


def _fmt_plain(val: float, decimals: int = 1) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Markdown section builder
# ---------------------------------------------------------------------------
def _overall_table(trades: pd.DataFrame, label: str) -> list[str]:
    lines: list[str] = []
    lines.append(f"**{label} — Overall**")
    lines.append("")
    lines.append("| Dir filled | N signals | Fill% | Win% | Avg win (pts) | Avg loss (pts) | Expectancy | Sharpe |")
    lines.append("|-----------|-----------|-------|------|---------------|----------------|------------|--------|")
    for direction in ["long", "short", "all"]:
        if direction == "all":
            sub = trades
        else:
            sub = trades[trades["filled_direction"] == direction]
        m = _metrics(sub)
        if not m or m["n_filled"] < 5:
            continue
        lines.append(
            f"| {direction} | {m['n_signals']} | {_fmt_plain(m['fill_rate_pct'])}% "
            f"| {_fmt_plain(m['win_rate_pct'])}% | {_fmt(m['avg_win_pts'])} "
            f"| {_fmt(m['avg_loss_pts'])} | {_fmt(m['expectancy_pts'])} | {_fmt(m['sharpe'], 3)} |"
        )
    lines.append("")
    return lines


def _segmented_table(trades: pd.DataFrame, seg_col: str, seg_vals: list, label: str) -> list[str]:
    lines: list[str] = [f"**{label} — by {seg_col}**", ""]
    lines.append(f"| {seg_col} | N filled | Win% | Avg win | Avg loss | Expectancy |")
    lines.append(f"|{'-' * (len(seg_col) + 2)}|----------|------|---------|----------|------------|")
    for val in seg_vals:
        sub = trades[trades[seg_col] == val]
        m = _metrics(sub)
        if not m or m["n_filled"] < 5:
            continue
        lines.append(
            f"| {val} | {m['n_filled']} | {_fmt_plain(m['win_rate_pct'])}% "
            f"| {_fmt(m['avg_win_pts'])} | {_fmt(m['avg_loss_pts'])} | {_fmt(m['expectancy_pts'])} |"
        )
    lines.append("")
    return lines


def build_section(asrs: pd.DataFrame, srs: pd.DataFrame) -> str:
    # Annotate range buckets
    range_bins = [0, 10, 25, float("inf")]
    for df in [asrs, srs]:
        df["range_cat"] = pd.cut(
            df["bar_range"], bins=range_bins, labels=RANGE_LABELS, right=False
        )
        df["weekday"] = pd.Categorical(df["weekday"], categories=WEEKDAY_ORDER, ordered=True)

    lines: list[str] = [
        "",
        "---",
        "",
        "## Hougaard DAX Bracket Strategies — IBDE40",
        "",
        f"Generated: {date.today().isoformat()}  ",
        "Instrument: IBDE40, Frankfurt time  ",
        "Period: 2020-01-01 -> 2026-01-01  ",
        "Entry: OCO bracket ±2 pts of signal bar high/low  ",
        "Exit: 2-bar trailing stop or EOD (17:30 Frankfurt)  ",
        "Cost: 2 pts round-trip spread  ",
        "",
        "---",
        "",
        "### BT-4-S2: ASRS (4th 5-min bar at 09:15 Frankfurt)",
        "",
    ]

    lines += _overall_table(asrs, "ASRS")
    lines += _segmented_table(asrs, "range_cat", RANGE_LABELS, "ASRS")
    lines += _segmented_table(asrs, "weekday", WEEKDAY_ORDER, "ASRS")

    lines += [
        "---",
        "",
        "### BT-4-S3: SRS (2nd 15-min bar at 09:15 Frankfurt)",
        "",
    ]
    lines += _overall_table(srs, "SRS")
    lines += _segmented_table(srs, "range_cat", RANGE_LABELS, "SRS")
    lines += _segmented_table(srs, "weekday", WEEKDAY_ORDER, "SRS")

    # Comparison
    asrs_m = _metrics(asrs)
    srs_m = _metrics(srs)
    lines += [
        "---",
        "",
        "### ASRS vs SRS Comparison",
        "",
        "| Strategy | N filled | Win% | Expectancy (pts) | Sharpe | Go/No-go |",
        "|----------|----------|------|-----------------|--------|----------|",
    ]
    for label, m in [("ASRS (4th 5m)", asrs_m), ("SRS (2nd 15m)", srs_m)]:
        if not m:
            continue
        verdict = "Go" if m["expectancy_pts"] > 0 else "No-go"
        lines.append(
            f"| {label} | {m['n_filled']} | {_fmt_plain(m['win_rate_pct'])}% "
            f"| {_fmt(m['expectancy_pts'])} | {_fmt(m['sharpe'], 3)} | {verdict} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading bars...")
    df5 = load_bars("5min")
    df15 = load_bars("15min")
    print(f"  5-min: {len(df5):,} bars | 15-min: {len(df15):,} bars")

    print("Loading ATR...")
    atr = load_daily_atr()

    print("Running ASRS (BT-4-S2)...")
    asrs = run_asrs(df5, atr)
    filled = asrs["filled_direction"].notna().sum()
    print(f"  {len(asrs)} signal days, {filled} filled ({filled/len(asrs)*100:.1f}%)")

    print("Running SRS (BT-4-S3)...")
    srs = run_srs(df15, atr)
    filled = srs["filled_direction"].notna().sum()
    print(f"  {len(srs)} signal days, {filled} filled ({filled/len(srs)*100:.1f}%)")

    section = build_section(asrs, srs)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"Appended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
