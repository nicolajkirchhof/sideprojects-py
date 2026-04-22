"""
forex_oco_bt.py
===============
OCO candle scan for forex session opens.

Adapts the equity-index candle scan (candle_scan.py) to forex pairs.
Tests every (pair × session × timeframe × bar_index × stop_method) combination
within the first 2 hours of each session open.

Key differences from equity scan
---------------------------------
- No exchange auction: sessions defined by UTC institutional open times.
- Price units: all results reported in pips (price_move / pip_size).
- Cost: SPREAD_COST_PIPS round-trip (tighter than equity CFD 2-pt cost).
- Entry offset: ENTRY_OFFSET_PIPS beyond bar high/low.
- Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD.
- Stop methods: atr (SRS-style ATR trail), bar_range (ASRS-style OCO).

Sessions (UTC)
--------------
  London open  07:00 UTC  — EURUSD, GBPUSD, USDJPY
  NY open      13:30 UTC  — EURUSD, GBPUSD
  Tokyo open   00:00 UTC  — USDJPY, AUDUSD
  Sydney open  22:00 UTC  — AUDUSD (previous calendar day in UTC)

Run from repo root:
    python finance/intraday_pm/forex/forex_oco_bt.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

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
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCHEMA = "forex"
START  = datetime(2020, 1, 1, tzinfo=timezone.utc)
END    = datetime(2026, 4, 1, tzinfo=timezone.utc)

# Price move per 1 pip for each pair
PIP_SIZE: dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "CHFUSD": 0.0001,
    "USDJPY": 0.01,
}

# Costs and offset in pips (converted to price units per pair at runtime)
SPREAD_COST_PIPS  = 1.5
ENTRY_OFFSET_PIPS = 0.5

TIMEFRAMES     = ["5min", "15min", "30min"]
STOP_METHODS   = ["atr", "bar_range"]
MAX_SCAN_HOURS = 2

TF_MINUTES: dict[str, int] = {"5min": 5, "15min": 15, "30min": 30}

# Session opens: {name: (utc_hour, utc_minute)}
SESSIONS: dict[str, tuple[int, int]] = {
    "london": (7,  0),
    "ny":     (13, 30),
    "tokyo":  (0,  0),
    "sydney": (22, 0),
}

# Sessions to scan per pair
PAIR_SESSIONS: dict[str, list[str]] = {
    "EURUSD": ["london", "ny"],
    "GBPUSD": ["london", "ny"],
    "USDJPY": ["tokyo",  "london"],
    "AUDUSD": ["sydney", "tokyo"],
}

RESULTS_APPEND_PATH = "finance/intraday_pm/forex/RESULTS.md"
TOP_N = 10  # top combinations to show per pair × session


# ---------------------------------------------------------------------------
# Public helpers (imported by tests)
# ---------------------------------------------------------------------------
def _to_pips(price_move: float, pip_size: float) -> float:
    """Convert a price-unit move to pips."""
    return price_move / pip_size


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _tf_minutes(tf: str) -> int:
    return TF_MINUTES[tf]


def _max_bar_idx(tf: str) -> int:
    return (MAX_SCAN_HOURS * 60) // _tf_minutes(tf)


def _bar_time_utc(session: str, bar_idx: int, tf: str) -> str:
    open_h, open_m = SESSIONS[session]
    total_min = open_h * 60 + open_m + bar_idx * _tf_minutes(tf)
    return f"{total_min // 60:02d}:{total_min % 60:02d} UTC"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_bars(pair: str, tf: str) -> pd.DataFrame:
    """Load and resample forex bars; index stays UTC."""
    df = get_bars(pair, SCHEMA, START, END, period=tf)
    if df.empty:
        raise RuntimeError(f"No {tf} data for {pair}")
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})


def _load_daily_atr(pair: str) -> pd.Series:
    """ATR-14 from daily bars. Returns Series indexed by date."""
    df = get_bars(pair, SCHEMA, START, END, period="1D")
    df = df.rename(columns={"h": "high", "l": "low", "c": "close"})
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    atr.index = atr.index.date
    return atr


# ---------------------------------------------------------------------------
# Session bars for a given calendar date and session window
# ---------------------------------------------------------------------------
def _get_session_bars(
    df_tf: pd.DataFrame,
    session: str,
    calendar_date: date,
) -> pd.DataFrame:
    """
    Return bars starting at session open on calendar_date.

    The Sydney session (22:00 UTC) starts on the *previous* UTC calendar day,
    so we look for bars starting at 22:00 on calendar_date - 1 day.
    For all other sessions the session open is on calendar_date itself.
    """
    open_h, open_m = SESSIONS[session]

    if session == "sydney":
        base_date = calendar_date - timedelta(days=1)
    else:
        base_date = calendar_date

    session_open_ts = pd.Timestamp(base_date).tz_localize("UTC").replace(
        hour=open_h, minute=open_m, second=0
    )
    session_end_ts  = pd.Timestamp(calendar_date).tz_localize("UTC").replace(
        hour=23, minute=59, second=59
    )

    mask = (df_tf.index >= session_open_ts) & (df_tf.index <= session_end_ts)
    return df_tf[mask]


# ---------------------------------------------------------------------------
# Run scan for one (pair, session, tf, stop_method)
# ---------------------------------------------------------------------------
def _run_combo(
    pair: str,
    session: str,
    tf: str,
    stop_method: str,
    df_tf: pd.DataFrame,
    atr_by_date: pd.Series,
) -> pd.DataFrame:
    """
    Scan all bar indices across all trading days.
    Returns a summary DataFrame with one row per bar_idx, sorted by Sharpe.
    """
    pip_size      = PIP_SIZE[pair]
    entry_offset  = ENTRY_OFFSET_PIPS * pip_size
    spread_cost   = SPREAD_COST_PIPS  * pip_size
    max_idx       = _max_bar_idx(tf)

    # Collect all trading dates
    trading_dates = sorted({
        d for d in df_tf.index.date
        if pd.Timestamp(d).weekday() < 5
    })

    # Build sessions dict: {date: session_bars_df}
    sessions: dict[date, pd.DataFrame] = {}
    for cal_date in trading_dates:
        sess = _get_session_bars(df_tf, session, cal_date)
        if len(sess) >= 2:
            sessions[cal_date] = sess

    rows: list[dict] = []

    for bar_idx in range(max_idx):
        records: list[dict] = []

        for cal_date, session_bars in sessions.items():
            if len(session_bars) <= bar_idx:
                continue

            signal_bar = session_bars.iloc[bar_idx]
            remaining  = session_bars.iloc[bar_idx + 1:]
            if remaining.empty:
                continue

            atr_val       = atr_by_date.get(cal_date, float("nan"))
            bar_range_pts = signal_bar["high"] - signal_bar["low"] + 2 * entry_offset

            if stop_method == "atr":
                stop_pts  = atr_val * ATR_TRAIL_FACTOR if not np.isnan(atr_val) else bar_range_pts
                exit_mode = "2bar_trail"
            elif stop_method == "bar_range":
                stop_pts  = bar_range_pts
                exit_mode = "2bar_trail"
            else:
                raise ValueError(f"Unknown stop_method: {stop_method!r}")

            rec = _simulate_oco(
                signal_bar, remaining, stop_pts,
                exit_mode=exit_mode,
                entry_offset=entry_offset,
                spread_cost=spread_cost,
            )
            rec["date"] = cal_date
            records.append(rec)

        if not records:
            continue

        m = _metrics(pd.DataFrame(records))
        if not m or m.get("n_filled", 0) < 10:
            continue

        ev_pips       = _to_pips(m["expectancy_pts"], pip_size)
        avg_win_pips  = _to_pips(m["avg_win_pts"],    pip_size)
        avg_loss_pips = _to_pips(m["avg_loss_pts"],   pip_size)

        rows.append({
            "pair":         pair,
            "session":      session,
            "tf":           tf,
            "stop_method":  stop_method,
            "bar_idx":      bar_idx,
            "bar_time":     _bar_time_utc(session, bar_idx, tf),
            "n_signals":    m["n_signals"],
            "n_filled":     m["n_filled"],
            "win_pct":      m["win_rate_pct"],
            "avg_win_pip":  avg_win_pips,
            "avg_loss_pip": avg_loss_pips,
            "ev_pip":       ev_pips,
            "sharpe":       m["sharpe"],
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.sort_values("sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------
def _build_section(all_results: list[pd.DataFrame]) -> str:
    lines = [
        "",
        "---",
        "",
        "## Forex OCO Candle Scan",
        "",
        f"Generated: {date.today().isoformat()}  ",
        f"Pairs: {', '.join(PAIR_SESSIONS)}  ",
        f"Period: {START.date().isoformat()} -> {END.date().isoformat()}  ",
        f"Cost: {SPREAD_COST_PIPS} pips round-trip  ",
        f"Entry offset: {ENTRY_OFFSET_PIPS} pip  ",
        "Sessions: London 07:00, NY 13:30, Tokyo 00:00, Sydney 22:00 UTC  ",
        "",
        "### Results per pair × session",
        "",
    ]

    combined = (
        pd.concat(all_results, ignore_index=True)
        if all_results else pd.DataFrame()
    )

    for pair in PAIR_SESSIONS:
        for session in PAIR_SESSIONS[pair]:
            subset = (
                combined[
                    (combined["pair"] == pair) & (combined["session"] == session)
                ]
                if not combined.empty else pd.DataFrame()
            )

            lines.append(f"#### {pair} — {session.capitalize()} open")
            lines.append("")

            if subset.empty:
                lines.append("_No results_")
                lines.append("")
                continue

            positive = subset[subset["ev_pip"] > 0]
            lines.append(
                f"Positive EV combinations: {len(positive)} / {len(subset)}  "
            )
            if not positive.empty:
                best = positive.iloc[0]
                lines.append(
                    f"Best Sharpe: {best['tf']}/bar{int(best['bar_idx'])} "
                    f"{best['stop_method']} @ {best['bar_time']} "
                    f"-> Sharpe {_fmt(best['sharpe'], 3)}, "
                    f"EV {_fmt(best['ev_pip'])} pips  "
                )
            lines += [
                "",
                "| TF | Bar | Stop | Time (UTC) | N fills | Win% | Avg win (pip) "
                "| Avg loss (pip) | EV (pip) | Sharpe |",
                "|----|-----|------|------------|---------|------|---------------|"
                "----------------|----------|--------|",
            ]
            for _, row in subset.head(TOP_N).iterrows():
                lines.append(
                    f"| {row['tf']} | bar{int(row['bar_idx'])} | {row['stop_method']} "
                    f"| {row['bar_time']} | {int(row['n_filled'])} "
                    f"| {_fmt_plain(row['win_pct'])}% "
                    f"| {_fmt(row['avg_win_pip'])} | {_fmt(row['avg_loss_pip'])} "
                    f"| {_fmt(row['ev_pip'])} | {_fmt(row['sharpe'], 3)} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    all_results: list[pd.DataFrame] = []

    for pair in PAIR_SESSIONS:
        print(f"\n{pair}")
        atr_by_date = _load_daily_atr(pair)

        for session in PAIR_SESSIONS[pair]:
            print(f"  session={session}")

            for tf in TIMEFRAMES:
                df_tf = _load_bars(pair, tf)

                for stop_method in STOP_METHODS:
                    agg = _run_combo(pair, session, tf, stop_method, df_tf, atr_by_date)
                    if not agg.empty:
                        all_results.append(agg)

        # Quick per-pair summary
        pair_frames = [r for r in all_results if not r.empty and (r["pair"] == pair).any()]
        if pair_frames:
            pair_df  = pd.concat(pair_frames, ignore_index=True)
            positive = pair_df[pair_df["ev_pip"] > 0]
            best     = pair_df.sort_values("sharpe", ascending=False).iloc[0]
            print(
                f"  Positive EV: {len(positive)}/{len(pair_df)}  "
                f"Best: {best['tf']}/bar{int(best['bar_idx'])} {best['stop_method']} "
                f"{best['session']} -> Sharpe {best['sharpe']:+.3f}, "
                f"EV {best['ev_pip']:+.2f} pips"
            )

    section = _build_section(all_results)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"\nAppended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
