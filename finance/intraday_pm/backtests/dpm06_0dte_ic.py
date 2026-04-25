"""
dpm06_0dte_ic.py
================
DPM-06: 0DTE Iron Condor on SPX (IBUS500 CFD proxy).

Strategy:
  Sell an iron condor daily at approximately 20-delta short strikes.
  Wing protection at 2x the expected daily move.
  Hold to expiry (end of day). No active management.

IV proxy: HV20 (20-day rolling annualized realized vol from daily closes).
Regime proxy: rolling 252-day percentile of HV20 (VIX not available in stack).

Segmentation:
  - Day of week (Mon-Fri)
  - Vol regime: low (<33rd pct), mid (33-67th pct), high (>67th pct)

Cost model: 20% of credit received as bid-ask friction (options cost model).

Run from repo root:
    python finance/intraday_pm/backtests/dpm06_0dte_ic.py
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
from finance.core_pm.backtest import (
    BacktestConfig,
    simulate_trades,
    trades_to_dataframe,
    bs_put_strike,
    bs_call_strike,
    bs_put_price,
    bs_call_price,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL   = "IBUS500"
SCHEMA   = "cfd"
ET_TZ    = ZoneInfo("America/New_York")
START    = datetime(2020, 1, 1, tzinfo=timezone.utc)
END      = datetime(2026, 4, 1, tzinfo=timezone.utc)

HV_PERIOD       = 20    # days for realized vol computation
HV_PCT_WINDOW   = 252   # rolling window for percentile rank
MA200_PERIOD    = 200
TRADING_DAYS    = 252

DELTA_SHORT     = 0.20  # short strike delta (~20 delta = ~1.0 sigma in 1 day)
IC_WING_MULT    = 2.0   # long strike placed at WING_MULT x expected daily move from spot
COST_FRACTION   = 0.20  # 20% of credit as friction

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"

WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
REGIME_ORDER  = ["low_iv", "mid_iv", "high_iv"]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_daily(symbol: str = SYMBOL) -> pd.DataFrame:
    df = get_bars(symbol, SCHEMA, START, END, period="1D")
    if df.empty:
        raise RuntimeError(f"No daily data for {symbol}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df = df[["open", "high", "low", "close"]].copy()

    # HV20 as IV proxy (annualized)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["ivc"] = log_ret.rolling(HV_PERIOD).std() * np.sqrt(TRADING_DAYS)

    # Rolling percentile of HV20 (proxy for IVP / VIX regime)
    df["iv_pct"] = (
        df["ivc"]
        .rolling(HV_PCT_WINDOW, min_periods=HV_PERIOD)
        .rank(pct=True) * 100
    )

    # 200-day SMA
    df["ma200"] = df["close"].rolling(MA200_PERIOD).mean()

    # Rename close to 'c' for the core_pm engine
    df = df.rename(columns={"close": "c"})
    df.attrs["symbol"] = symbol

    return df.dropna(subset=["ivc"])


# ---------------------------------------------------------------------------
# Override ic_wing_width dynamically per trade
# We simulate manually to allow dynamic wing width (core_pm uses a fixed %).
# ---------------------------------------------------------------------------
def simulate_0dte(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate 0DTE IC (hold to expiry) with dynamically scaled wings.
    Short strikes at DELTA_SHORT delta; long strikes at IC_WING_MULT x daily expected move.
    """
    prices  = df["c"].values
    ivs     = df["ivc"].values
    iv_pcts = df["iv_pct"].values
    dates   = df.index
    n       = len(df)
    records = []

    for i in range(n - 1):
        iv = ivs[i]
        if np.isnan(iv) or iv <= 0:
            continue

        spot = prices[i]
        dte  = 1

        # Short strikes (~20 delta)
        put_short  = bs_put_strike(spot, iv, dte, DELTA_SHORT)
        call_short = bs_call_strike(spot, iv, dte, DELTA_SHORT)

        # Long strikes (protection at IC_WING_MULT x expected daily move)
        daily_move = spot * iv * np.sqrt(1.0 / TRADING_DAYS)
        put_long   = spot - IC_WING_MULT * daily_move
        call_long  = spot + IC_WING_MULT * daily_move

        # Credit collected (short side only — long cost negligible far OTM)
        put_credit  = bs_put_price(spot, put_short, iv, dte)
        call_credit = bs_call_price(spot, call_short, iv, dte)
        gross_credit = put_credit + call_credit
        net_credit   = gross_credit * (1 - COST_FRACTION)

        if net_credit <= 0:
            continue

        # P&L at expiry
        exit_price = prices[i + 1]

        put_intrinsic  = max(put_short  - exit_price, 0.0)
        call_intrinsic = max(exit_price - call_short, 0.0)
        put_wing_cap   = max(put_short  - put_long,   0.0)
        call_wing_cap  = max(call_long  - call_short, 0.0)

        put_loss   = min(put_intrinsic,  put_wing_cap)
        call_loss  = min(call_intrinsic, call_wing_cap)
        pnl        = net_credit - (put_loss + call_loss)

        regime = (
            "low_iv"  if iv_pcts[i] < 33 else
            "high_iv" if iv_pcts[i] > 67 else
            "mid_iv"
        )

        records.append({
            "date"          : dates[i],
            "weekday"       : dates[i].weekday(),
            "weekday_name"  : WEEKDAY_NAMES.get(dates[i].weekday(), "?"),
            "spot"          : spot,
            "iv"            : iv,
            "iv_pct"        : iv_pcts[i],
            "regime"        : regime,
            "put_short"     : put_short,
            "call_short"    : call_short,
            "put_long"      : put_long,
            "call_long"     : call_long,
            "net_credit"    : net_credit,
            "exit_price"    : exit_price,
            "pnl"           : pnl,
            "win"           : pnl > 0,
            "put_breached"  : exit_price < put_short,
            "call_breached" : exit_price > call_short,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _seg_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    n      = len(df)
    wins   = df["win"].sum()
    losses = n - wins
    avg_w  = df.loc[df["win"],  "pnl"].mean() if wins   else 0.0
    avg_l  = df.loc[~df["win"], "pnl"].mean() if losses else 0.0
    ev     = df["pnl"].mean()
    std    = df["pnl"].std()
    sharpe = ev / std * np.sqrt(TRADING_DAYS) if std > 0 else 0.0
    return dict(n=n, win_pct=wins/n*100, avg_win=avg_w, avg_loss=avg_l,
                ev=ev, sharpe=sharpe, total=df["pnl"].sum())


def _fmt(v: float, dp: int = 2) -> str:
    return f"{v:+.{dp}f}"


def print_report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("  No trades.")
        return

    m = _seg_metrics(trades)
    sep = "-" * 62

    print(f"\n{sep}")
    print(f"  DPM-06: 0DTE Iron Condor  --  {SYMBOL}  {START.date()} to {END.date()}")
    print(f"  HV20 IV proxy  |  {DELTA_SHORT:.0%} delta shorts  |  {COST_FRACTION:.0%} cost friction")
    print(sep)
    print(f"  Trades     : {m['n']}")
    print(f"  Win rate   : {m['win_pct']:.1f}%")
    print(f"  Avg win    : {_fmt(m['avg_win'])} pts")
    print(f"  Avg loss   : {_fmt(m['avg_loss'])} pts")
    print(f"  Expectancy : {_fmt(m['ev'])} pts/trade")
    print(f"  Sharpe     : {_fmt(m['sharpe'], 3)}")
    print(f"  Total P&L  : {_fmt(m['total'])} pts")

    breaches = trades["put_breached"].sum() + trades["call_breached"].sum()
    print(f"  Breaches   : {breaches} ({breaches/m['n']*100:.1f}%)")
    print()

    # By day of week
    print("  By day of week:")
    print(f"  {'Day':<6}  {'N':>5}  {'Win%':>6}  {'EV':>8}  {'Sharpe':>7}")
    for wd in range(5):
        sub = trades[trades["weekday"] == wd]
        if sub.empty:
            continue
        sm = _seg_metrics(sub)
        name = WEEKDAY_NAMES[wd]
        print(f"  {name:<6}  {sm['n']:>5}  {sm['win_pct']:>5.1f}%  {_fmt(sm['ev']):>8}  {_fmt(sm['sharpe'], 3):>7}")
    print()

    # By vol regime
    print("  By HV regime (VIX proxy):")
    print(f"  {'Regime':<12}  {'N':>5}  {'Win%':>6}  {'EV':>8}  {'Sharpe':>7}")
    for reg in REGIME_ORDER:
        sub = trades[trades["regime"] == reg]
        if sub.empty:
            continue
        sm = _seg_metrics(sub)
        print(f"  {reg:<12}  {sm['n']:>5}  {sm['win_pct']:>5.1f}%  {_fmt(sm['ev']):>8}  {_fmt(sm['sharpe'], 3):>7}")
    print()


def build_section(trades: pd.DataFrame) -> str:
    if trades.empty:
        return ""
    m = _seg_metrics(trades)
    lines = [
        "",
        "---",
        "",
        f"## DPM-06: 0DTE Iron Condor -- {SYMBOL}",
        "",
        f"Generated: {pd.Timestamp.today().date()}  ",
        f"Period: {START.date()} to {END.date()}  ",
        f"IV proxy: HV20 (annualized realized vol, {HV_PERIOD}-day)  ",
        f"Short strikes: {DELTA_SHORT:.0%} delta  |  Cost friction: {COST_FRACTION:.0%} of credit  ",
        "",
        "| Segment | N | Win% | Avg win | Avg loss | EV (pts) | Sharpe |",
        "|---------|---|------|---------|----------|----------|--------|",
        f"| Overall | {m['n']} | {m['win_pct']:.1f}% | {_fmt(m['avg_win'])} | {_fmt(m['avg_loss'])} | {_fmt(m['ev'])} | {_fmt(m['sharpe'], 3)} |",
    ]

    for wd in range(5):
        sub = trades[trades["weekday"] == wd]
        if sub.empty:
            continue
        sm = _seg_metrics(sub)
        name = WEEKDAY_NAMES[wd]
        lines.append(
            f"| {name} | {sm['n']} | {sm['win_pct']:.1f}% | {_fmt(sm['avg_win'])} | {_fmt(sm['avg_loss'])} | {_fmt(sm['ev'])} | {_fmt(sm['sharpe'], 3)} |"
        )

    for reg in REGIME_ORDER:
        sub = trades[trades["regime"] == reg]
        if sub.empty:
            continue
        sm = _seg_metrics(sub)
        lines.append(
            f"| {reg} | {sm['n']} | {sm['win_pct']:.1f}% | {_fmt(sm['avg_win'])} | {_fmt(sm['avg_loss'])} | {_fmt(sm['ev'])} | {_fmt(sm['sharpe'], 3)} |"
        )

    verdict = "Go" if m["ev"] > 0 and m["sharpe"] > 0.1 else "No-go"
    lines += [
        "",
        f"**Verdict: {verdict}**",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading daily bars for {SYMBOL}...")
    df = load_daily()
    print(f"  {len(df):,} days  ({df.index[0].date()} to {df.index[-1].date()})")

    print("Simulating 0DTE IC...")
    trades = simulate_0dte(df)
    print(f"  {len(trades)} trades simulated")

    print_report(trades)

    section = build_section(trades)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"Appended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
