"""
micro_macro_trend_bt.py
=======================
PoC backtest: two-level trend classification on 5-min bars.

Micro-trend: candle-pair direction — HH+HL = +1, LH+LL = -1, else 0.
             ATR non-overlap filter (default 10%) removes noisy overlapping bars.

Macro-trend: accumulated micro-trend with retracement reset.
             When a counter-move retraces > 90% of the established macro range,
             the algorithm backtracks to the last swing extrema and resets to neutral.

MA filter:   EMA(20) slope — only take long macro flips when EMA is rising,
             only short macro flips when EMA is falling.

Entry:  close of bar where macro_trend flips direction.
Exit:   macro_trend reversal or session end (17:30 Frankfurt).
Cost:   2 pts round-trip spread.

Scope: IBDE40, 2024-01-01 → 2026-04-01 (proof of concept excerpt).

Run from repo root:
    python finance/intraday_pm/backtests/micro_macro_trend_bt.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from finance.utils.intraday import get_bars

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL   = "IBDE40"
SCHEMA   = "cfd"
TZ       = ZoneInfo("Europe/Berlin")
START    = datetime(2024, 1, 1, tzinfo=timezone.utc)
END      = datetime(2026, 4, 1, tzinfo=timezone.utc)

SESSION_OPEN_H   = 9
SESSION_OPEN_M   = 0
SESSION_CLOSE_H  = 17
SESSION_CLOSE_M  = 30

SPREAD_COST_PTS            = 2.0
EMA_PERIOD                 = 20
ATR_OVERLAP_THRESHOLD      = 0.10   # min non-overlap fraction for micro-trend
MACRO_RETRACEMENT          = 0.90   # counter-move fraction triggering macro reset
PROFIT_TARGET_PTS          = 0.0    # fixed profit target (0 = disabled, ride to reversal/EOD)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_5min() -> pd.DataFrame:
    df = get_bars(SYMBOL, SCHEMA, START, END, period="5min")
    if df.empty:
        raise RuntimeError(f"No 5-min data for {SYMBOL}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(TZ)
    return df[["open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Micro-trend (vectorized)
# ---------------------------------------------------------------------------
def compute_micro_trend(df: pd.DataFrame) -> pd.Series:
    """
    Candle-pair directional classifier.
    +1 when current bar has higher high AND higher low vs prior bar,
    with at least ATR_OVERLAP_THRESHOLD non-overlap fraction.
    -1 for lower high AND lower low with same filter. 0 otherwise.
    """
    h, l   = df["high"], df["low"]
    h1, l1 = h.shift(1), l.shift(1)
    atr    = (h1 - l1).clip(lower=1e-9)

    # downtrend: lower high, lower low; non-overlap = portion of prior bar below current high
    down = (h1 > h) & (l1 > l) & ((atr - (h - l1).clip(lower=0)) / atr > ATR_OVERLAP_THRESHOLD)
    # uptrend: higher high, higher low; non-overlap = portion of prior bar above current low
    up   = (h1 < h) & (l1 < l) & ((atr - (h1 - l).clip(lower=0)) / atr > ATR_OVERLAP_THRESHOLD)

    micro = pd.Series(0, index=df.index, dtype=int)
    micro[up]   =  1
    micro[down] = -1
    return micro


# ---------------------------------------------------------------------------
# Macro-trend (stateful — backtracking requires a loop)
# ---------------------------------------------------------------------------
def compute_macro_trend(df: pd.DataFrame, micro: pd.Series) -> pd.Series:
    """
    Forward-only macro trend accumulation (no backtracking).

    Accumulates micro-trend into a macro direction.
    When a counter-move retraces > MACRO_RETRACEMENT of the established macro range,
    reset to neutral at the current bar and start fresh — range resets to current bar only.
    No retroactive relabelling; labels are assigned only with information available at each bar.
    """
    h = df["high"].values
    l = df["low"].values
    m = micro.values
    n = len(df)

    macro_series = np.zeros(n, dtype=int)
    macro_trend  = 0
    macro_high   = h[0]
    macro_low    = l[0]

    for i in range(1, n):
        mi = int(m[i])

        # extend macro range with every new bar
        macro_high = max(macro_high, h[i])
        macro_low  = min(macro_low,  l[i])

        if mi == macro_trend or macro_trend == 0 or mi == 0:
            macro_trend = mi if mi != 0 else macro_trend
        else:
            # counter-move — measure retracement vs established macro range
            macro_range = macro_high - macro_low
            if macro_range > 1e-9:
                retrace = (macro_high - l[i]) / macro_range if macro_trend == 1 \
                          else (h[i] - macro_low) / macro_range
                if retrace > MACRO_RETRACEMENT:
                    # significant retracement: reset state, start range from this bar
                    macro_trend = 0
                    macro_high  = h[i]
                    macro_low   = l[i]
                # below threshold: hold current macro direction, ignore noise

        macro_series[i] = macro_trend

    return pd.Series(macro_series, index=df.index, dtype=int)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute micro/macro trend and EMA(20) slope filter.
    Signal +1: macro flips to uptrend AND EMA is rising.
    Signal -1: macro flips to downtrend AND EMA is falling.
    """
    df = df.copy()
    micro = compute_micro_trend(df)
    macro = compute_macro_trend(df, micro)
    ema   = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    slope = ema - ema.shift(1)

    prev = macro.shift(1).fillna(0).astype(int)
    df["macro"] = macro
    df["slope"] = slope
    df["signal"] = 0
    df.loc[(macro ==  1) & (prev !=  1) & (slope > 0), "signal"] =  1
    df.loc[(macro == -1) & (prev != -1) & (slope < 0), "signal"] = -1
    return df


# ---------------------------------------------------------------------------
# Trade simulation (per session)
# ---------------------------------------------------------------------------
def simulate_session(df: pd.DataFrame) -> list[dict]:
    """
    Enter on close of bar where signal fires; exit on macro reversal or last bar.
    Only one trade at a time; a new signal while in-trade is ignored.
    """
    records: list[dict] = []
    in_trade  = False
    direction = 0
    entry_px  = 0.0
    last_ts   = df.index[-1]

    for ts, row in df.iterrows():
        is_eod = ts == last_ts

        if in_trade:
            unrealised = direction * (row["close"] - entry_px)
            hit_target = PROFIT_TARGET_PTS > 0 and unrealised >= PROFIT_TARGET_PTS
            reversal   = (direction == 1 and row["macro"] == -1) or \
                         (direction == -1 and row["macro"] == 1)
            if hit_target or reversal or is_eod:
                pnl = unrealised - SPREAD_COST_PTS
                reason = "target" if hit_target else ("reversal" if reversal else "eod")
                records.append({
                    "date"      : ts.date(),
                    "direction" : direction,
                    "entry_px"  : entry_px,
                    "exit_px"   : row["close"],
                    "pnl"       : pnl,
                    "reason"    : reason,
                })
                in_trade = False

        if not in_trade and row["signal"] != 0 and not is_eod:
            in_trade  = True
            direction = int(row["signal"])
            entry_px  = row["close"]

    return records


# ---------------------------------------------------------------------------
# Metrics report
# ---------------------------------------------------------------------------
def print_metrics(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("  No trades generated.")
        return

    n      = len(trades)
    wins   = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    longs  = trades[trades["direction"] ==  1]
    shorts = trades[trades["direction"] == -1]

    w_rate = len(wins) / n * 100
    avg_w  = wins["pnl"].mean()   if len(wins)   else 0.0
    avg_l  = losses["pnl"].mean() if len(losses) else 0.0
    ev     = trades["pnl"].mean()
    std    = trades["pnl"].std()
    sharpe = ev / std * np.sqrt(252) if std > 0 else 0.0

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Micro/Macro Trend PoC  --  {SYMBOL}  {START.date()} to {END.date()}")
    pt_str = f"{PROFIT_TARGET_PTS} pts" if PROFIT_TARGET_PTS > 0 else "off"
    print(f"  EMA({EMA_PERIOD}) slope filter  |  spread {SPREAD_COST_PTS} pts  |  ATR thr {ATR_OVERLAP_THRESHOLD}  |  target {pt_str}")
    print(sep)
    print(f"  Trades         :  {n}  (long {len(longs)}  /  short {len(shorts)})")
    print(f"  Win rate       :  {w_rate:.1f}%")
    print(f"  Avg win        :  {avg_w:+.2f} pts")
    print(f"  Avg loss       :  {avg_l:+.2f} pts")
    wl_ratio = abs(avg_w / avg_l) if avg_l != 0 else float("nan")
    print(f"  Win/loss ratio :  {wl_ratio:.2f}")
    print(f"  Expectancy     :  {ev:+.2f} pts/trade")
    print(f"  Sharpe (ann.)  :  {sharpe:+.3f}")
    print(f"  Total P&L      :  {trades['pnl'].sum():+.1f} pts")
    print(sep)

    by_reason = trades.groupby("reason")["pnl"].agg(count="count", avg="mean", total="sum")
    print("\n  By exit reason:")
    for reason, row in by_reason.iterrows():
        print(f"    {reason:12s}  n={int(row['count']):4d}  avg={row['avg']:+.2f}  total={row['total']:+.1f}")

    by_dir = trades.groupby("direction")["pnl"].agg(count="count", avg="mean", total="sum")
    print("\n  By direction:")
    for d, row in by_dir.iterrows():
        label = "long " if d == 1 else "short"
        print(f"    {label}        n={int(row['count']):4d}  avg={row['avg']:+.2f}  total={row['total']:+.1f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading 5-min bars for {SYMBOL}...")
    df = load_5min()
    print(f"  {len(df):,} bars  ({df.index[0].date()} to {df.index[-1].date()})")

    # Clip to session hours (09:00–17:30 Frankfurt)
    session_mask = (
        (df.index.hour > SESSION_OPEN_H) |
        ((df.index.hour == SESSION_OPEN_H) & (df.index.minute >= SESSION_OPEN_M))
    ) & (
        (df.index.hour < SESSION_CLOSE_H) |
        ((df.index.hour == SESSION_CLOSE_H) & (df.index.minute <= SESSION_CLOSE_M))
    )
    df = df[session_mask]
    print(f"  {len(df):,} session bars after filtering")

    print("Simulating per session...")
    all_records: list[dict] = []
    df["_date"] = df.index.date
    days_run = 0

    for _day, day_df in df.groupby("_date"):
        if len(day_df) < EMA_PERIOD + 5:
            continue
        signals = build_signals(day_df.drop(columns=["_date"]))
        records = simulate_session(signals)
        all_records.extend(records)
        days_run += 1

    print(f"  {days_run} sessions  |  {len(all_records)} trades")

    trades = pd.DataFrame(all_records)
    print_metrics(trades)


if __name__ == "__main__":
    main()
