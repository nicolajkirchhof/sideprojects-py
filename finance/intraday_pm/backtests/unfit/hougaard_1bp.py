"""
hougaard_1bp.py
===============
Backtest: Hougaard 1BP / 1BN strategy on IBGB100 (FTSE 100 futures), 5-min bars.

Strategy rules
--------------
Signal bar: the first 5-min bar of the London session (08:00 London time).

1BP (short signal):
  - Signal bar closes positive (close > open)
  - Entry: sell-stop order placed 2 pts below signal bar's low
  - Stop: signal bar's range (high - low) above entry price
  - Exit: 2-bar trailing stop (trail stop to max of last 2 completed bar highs)

1BN (long signal):
  - Signal bar closes negative (close < open)
  - Entry: buy-stop order placed 2 pts above signal bar's high
  - Stop: signal bar's range below entry price
  - Exit: 2-bar trailing stop (trail stop to min of last 2 completed bar lows)

Both entry stop orders expire at EOD (16:30 London time) if unfilled.
Trade is exited at EOD if still open.

Comparison: also test an ATR trailing stop (20% of 14-day daily ATR).
Segment results by daily ATR regime: <50 pts, 50-80 pts, >80 pts.
Cost: 2 pts round-trip spread deducted from each completed trade.

Run from repo root:
    python finance/intraday_pm/backtests/hougaard_1bp.py
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
SYMBOL = "IBGB100"
SCHEMA = "cfd"
START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 1, tzinfo=timezone.utc)

LONDON_TZ = ZoneInfo("Europe/London")
SESSION_OPEN_LOCAL = (8, 0)   # 08:00 London time
SESSION_CLOSE_LOCAL = (16, 30)  # 16:30 London time

ENTRY_OFFSET_PTS = 2.0   # pts beyond signal bar for stop order
SPREAD_COST_PTS = 2.0    # round-trip spread cost in pts
ATR_TRAIL_FACTOR = 0.20  # ATR trailing stop = 20% of 14-day ATR
ATR_PERIOD = 14          # days for ATR calculation

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_5min() -> pd.DataFrame:
    """Load IBGB100 5-min bars and convert index to London time."""
    df = get_bars(SYMBOL, SCHEMA, START, END, period="5min")
    if df.empty:
        raise RuntimeError(f"No 5-min data found for {SYMBOL}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(LONDON_TZ)
    return df[["open", "high", "low", "close"]]


def load_daily_atr() -> pd.Series:
    """Compute 14-day ATR from daily bars. Returns Series indexed by UTC date."""
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
# Single-day simulation
# ---------------------------------------------------------------------------
def _simulate_day(
    signal_bar: pd.Series,
    remaining_bars: pd.DataFrame,
    direction: str,  # "short" or "long"
    atr_pts: float,
) -> dict:
    """
    Simulate one trade for a single day.

    Parameters
    ----------
    signal_bar     : The 08:00 London bar (the 1BP/1BN bar).
    remaining_bars : All 5-min bars after the signal bar, up to 16:30.
    direction      : "short" for 1BP, "long" for 1BN.
    atr_pts        : 14-day ATR in points (for ATR trailing stop comparison).

    Returns
    -------
    dict with keys: date, direction, filled, result_2bar_pts, result_atr_pts,
    entry, stop_initial, win_2bar, win_atr.
    """
    bar_range = signal_bar["high"] - signal_bar["low"]

    if direction == "short":
        entry_level = signal_bar["low"] - ENTRY_OFFSET_PTS
        stop_initial = entry_level + bar_range
        atr_stop_dist = atr_pts * ATR_TRAIL_FACTOR
    else:
        entry_level = signal_bar["high"] + ENTRY_OFFSET_PTS
        stop_initial = entry_level - bar_range
        atr_stop_dist = atr_pts * ATR_TRAIL_FACTOR

    result_base = {
        "date": signal_bar.name.date(),
        "direction": direction,
        "filled": False,
        "result_2bar_pts": np.nan,
        "result_atr_pts": np.nan,
        "entry": entry_level,
        "stop_initial": stop_initial,
        "bar_range": bar_range,
        "win_2bar": np.nan,
        "win_atr": np.nan,
    }

    if remaining_bars.empty:
        return result_base

    # Check for entry fill: first bar that touches entry level
    entry_bar_idx = None
    for idx, bar in remaining_bars.iterrows():
        if direction == "short" and bar["low"] <= entry_level:
            entry_bar_idx = idx
            break
        if direction == "long" and bar["high"] >= entry_level:
            entry_bar_idx = idx
            break

    if entry_bar_idx is None:
        return result_base  # stop order never filled

    result_base["filled"] = True
    post_entry = remaining_bars.loc[remaining_bars.index > entry_bar_idx].copy()

    # ---- 2-bar trailing stop simulation ----
    stop_2bar = stop_initial
    exit_price_2bar = None
    bar_history: list[pd.Series] = []

    for idx, bar in post_entry.iterrows():
        # Check if stop is hit intrabar (use high/low)
        if direction == "short" and bar["high"] >= stop_2bar:
            exit_price_2bar = stop_2bar
            break
        if direction == "long" and bar["low"] <= stop_2bar:
            exit_price_2bar = stop_2bar
            break

        bar_history.append(bar)

        # Update 2-bar trailing stop after this bar completes
        if len(bar_history) >= 2:
            last_two = bar_history[-2:]
            if direction == "short":
                stop_2bar = min(stop_2bar, max(b["high"] for b in last_two) + 0.0)
                # Trail: stop = max of last 2 highs (move stop down as trade progresses)
                new_stop = max(b["high"] for b in last_two)
                # Only trail if it moves the stop in favour (down for short)
                if new_stop < stop_2bar:
                    stop_2bar = new_stop
            else:
                new_stop = min(b["low"] for b in last_two)
                if new_stop > stop_2bar:
                    stop_2bar = new_stop

    if exit_price_2bar is None:
        # EOD exit at last bar's close
        exit_price_2bar = post_entry.iloc[-1]["close"] if not post_entry.empty else entry_level

    if direction == "short":
        result_2bar_pts = (entry_level - exit_price_2bar) - SPREAD_COST_PTS
    else:
        result_2bar_pts = (exit_price_2bar - entry_level) - SPREAD_COST_PTS

    result_base["result_2bar_pts"] = result_2bar_pts
    result_base["win_2bar"] = result_2bar_pts > 0

    # ---- ATR trailing stop simulation ----
    if not np.isnan(atr_pts):
        stop_atr = stop_initial
        exit_price_atr = None

        for idx, bar in post_entry.iterrows():
            if direction == "short" and bar["high"] >= stop_atr:
                exit_price_atr = stop_atr
                break
            if direction == "long" and bar["low"] <= stop_atr:
                exit_price_atr = stop_atr
                break

            # Trail ATR stop: lock in profits as trade moves in our favour
            if direction == "short":
                candidate = bar["close"] + atr_stop_dist
                if candidate < stop_atr:
                    stop_atr = candidate
            else:
                candidate = bar["close"] - atr_stop_dist
                if candidate > stop_atr:
                    stop_atr = candidate

        if exit_price_atr is None:
            exit_price_atr = post_entry.iloc[-1]["close"] if not post_entry.empty else entry_level

        if direction == "short":
            result_atr_pts = (entry_level - exit_price_atr) - SPREAD_COST_PTS
        else:
            result_atr_pts = (exit_price_atr - entry_level) - SPREAD_COST_PTS

        result_base["result_atr_pts"] = result_atr_pts
        result_base["win_atr"] = result_atr_pts > 0

    return result_base


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------
def run_backtest(df5: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Iterate over each trading day and simulate 1BP + 1BN."""
    records: list[dict] = []
    df5["date"] = df5.index.date

    for trade_date, day_df in df5.groupby("date"):
        # Signal bar: first 5-min bar at 08:00 London time
        signal_mask = (day_df.index.hour == SESSION_OPEN_LOCAL[0]) & \
                      (day_df.index.minute == SESSION_OPEN_LOCAL[1])
        signal_bars = day_df[signal_mask]
        if signal_bars.empty:
            continue
        signal_bar = signal_bars.iloc[0]

        # Session close: 16:30 London time
        eod_mask = (day_df.index > signal_bar.name) & (
            (day_df.index.hour < SESSION_CLOSE_LOCAL[0]) |
            ((day_df.index.hour == SESSION_CLOSE_LOCAL[0]) &
             (day_df.index.minute <= SESSION_CLOSE_LOCAL[1]))
        )
        remaining = day_df[eod_mask]

        # Get ATR for this date (use prior day's ATR for the trade)
        atr_val = atr_series.get(trade_date, np.nan)

        # 1BP: signal bar positive → short
        if signal_bar["close"] > signal_bar["open"]:
            rec = _simulate_day(signal_bar, remaining, "short", atr_val)
            rec["signal"] = "1BP"
            records.append(rec)

        # 1BN: signal bar negative → long
        elif signal_bar["close"] < signal_bar["open"]:
            rec = _simulate_day(signal_bar, remaining, "long", atr_val)
            rec["signal"] = "1BN"
            records.append(rec)
        # Doji bar (close == open): skip

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(trades: pd.DataFrame, result_col: str) -> dict:
    filled = trades[trades["filled"]].copy()
    if filled.empty:
        return {}
    completed = filled[filled[result_col].notna()]
    if completed.empty:
        return {}
    n = len(completed)
    wins = completed[completed[result_col] > 0]
    losses = completed[completed[result_col] <= 0]
    win_rate = len(wins) / n * 100
    avg_win = wins[result_col].mean() if not wins.empty else 0.0
    avg_loss = losses[result_col].mean() if not losses.empty else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    total_pts = completed[result_col].sum()
    # Simple Sharpe: mean / std of trade results
    sharpe = completed[result_col].mean() / completed[result_col].std() if completed[result_col].std() > 0 else 0.0
    fill_rate = len(filled) / len(trades) * 100
    return {
        "n_signals": len(trades),
        "n_filled": n,
        "fill_rate_pct": fill_rate,
        "win_rate_pct": win_rate,
        "avg_win_pts": avg_win,
        "avg_loss_pts": avg_loss,
        "expectancy_pts": expectancy,
        "total_pts": total_pts,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# ATR regime segmentation
# ---------------------------------------------------------------------------
ATR_BINS = [0, 50, 80, float("inf")]
ATR_LABELS = ["<50 pts", "50-80 pts", ">80 pts"]


def _atr_regime(atr_val: float) -> str:
    if np.isnan(atr_val):
        return "unknown"
    if atr_val < 50:
        return "<50 pts"
    if atr_val <= 80:
        return "50-80 pts"
    return ">80 pts"


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------
def _fmt(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:+.{decimals}f}"


def _fmt_plain(val: float, decimals: int = 1) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"


def build_results_section(trades: pd.DataFrame) -> str:
    lines: list[str] = [
        "",
        "---",
        "",
        "## Hougaard 1BP/1BN — IBGB100 (FTSE 100)",
        "",
        f"Generated: {date.today().isoformat()}  ",
        "Instrument: IBGB100, 5-min bars  ",
        "Period: 2020-01-01 -> 2026-01-01  ",
        "Signal: first 5-min bar at 08:00 London time  ",
        "Entry: stop order 2 pts beyond signal bar high/low  ",
        "Stop: signal bar range (high - low)  ",
        "Exit: 2-bar trailing stop or EOD (16:30 London)  ",
        "Cost: 2 pts round-trip spread  ",
        "",
    ]

    # Overall by signal type and stop method
    lines.append("### Overall Results")
    lines.append("")
    lines.append("| Signal | Stop method | N signals | Fill% | Win% | Avg win (pts) | Avg loss (pts) | Expectancy (pts) | Sharpe |")
    lines.append("|--------|-------------|-----------|-------|------|---------------|----------------|-----------------|--------|")

    for signal in ["1BP", "1BN"]:
        sub = trades[trades["signal"] == signal]
        for col, label in [("result_2bar_pts", "2-bar trail"), ("result_atr_pts", "ATR trail (20%)")]:
            m = _metrics(sub, col)
            if not m:
                continue
            lines.append(
                f"| {signal} | {label} "
                f"| {m['n_signals']} | {_fmt_plain(m['fill_rate_pct'])}% "
                f"| {_fmt_plain(m['win_rate_pct'])}% "
                f"| {_fmt(m['avg_win_pts'])} | {_fmt(m['avg_loss_pts'])} "
                f"| {_fmt(m['expectancy_pts'])} | {_fmt(m['sharpe'], 3)} |"
            )
    lines.append("")

    # Segmented by daily ATR regime
    lines.append("### Results by Volatility Regime (14-day ATR)")
    lines.append("")
    lines.append("| Signal | Stop | ATR regime | N filled | Win% | Avg win | Avg loss | Expectancy |")
    lines.append("|--------|------|------------|----------|------|---------|----------|------------|")

    trades_filled = trades[trades["filled"]].copy()
    trades_filled["atr_regime"] = trades_filled["bar_range"].apply(
        lambda x: _atr_regime(x)  # use bar_range as proxy for intraday volatility
    )

    # Re-classify by actual bar range buckets
    trades_filled["atr_regime"] = pd.cut(
        trades_filled["bar_range"], bins=ATR_BINS, labels=ATR_LABELS, right=False
    )

    for signal in ["1BP", "1BN"]:
        for regime in ATR_LABELS:
            sub = trades_filled[
                (trades_filled["signal"] == signal) &
                (trades_filled["atr_regime"] == regime)
            ]
            for col, label in [("result_2bar_pts", "2-bar"), ("result_atr_pts", "ATR")]:
                m = _metrics(sub.assign(filled=True), col)
                if not m or m["n_filled"] < 5:
                    continue
                lines.append(
                    f"| {signal} | {label} | {regime} "
                    f"| {m['n_filled']} | {_fmt_plain(m['win_rate_pct'])}% "
                    f"| {_fmt(m['avg_win_pts'])} | {_fmt(m['avg_loss_pts'])} "
                    f"| {_fmt(m['expectancy_pts'])} |"
                )

    lines.append("")

    # Verdict
    lines += [
        "### Verdict",
        "",
    ]

    # Compute net positive cases
    verdicts = []
    for signal in ["1BP", "1BN"]:
        for col, label in [("result_2bar_pts", "2-bar trail"), ("result_atr_pts", "ATR trail")]:
            sub = trades[trades["signal"] == signal]
            m = _metrics(sub, col)
            if not m:
                continue
            ev = m["expectancy_pts"]
            verdict = "Go" if ev > 0 else "No-go"
            verdicts.append({"signal": signal, "stop": label, "ev": ev, "verdict": verdict})

    lines.append("| Signal | Stop method | Expectancy (pts) | Go / No-go |")
    lines.append("|--------|-------------|-----------------|------------|")
    for v in verdicts:
        lines.append(f"| {v['signal']} | {v['stop']} | {_fmt(v['ev'])} | {v['verdict']} |")

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading 5-min bars...")
    df5 = load_5min()
    print(f"  {len(df5):,} bars from {df5.index.min()} to {df5.index.max()}")

    print("Loading daily ATR...")
    atr_series = load_daily_atr()

    print("Running backtest...")
    trades = run_backtest(df5, atr_series)
    total = len(trades)
    filled = trades["filled"].sum()
    print(f"  {total} signal days, {filled} trades filled ({filled/total*100:.1f}%)")

    section = build_results_section(trades)

    # Append to RESULTS.md
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"Appended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
