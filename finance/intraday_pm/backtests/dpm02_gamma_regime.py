"""
dpm02_gamma_regime.py
=====================
DPM-02: Dealer Gamma Regime Filter.

GEX (Gamma Exposure) theory:
  Positive GEX: dealers are long gamma -> they sell rallies and buy dips
                -> dampened volatility -> intraday MEAN REVERSION
  Negative GEX: dealers are short gamma -> they chase moves
                -> amplified volatility -> intraday MOMENTUM

GEX is unavailable. Proxy: rolling 252-day percentile of HV20.
  Low HV (<33rd pct)  -> likely positive GEX -> expect mean reversion
  High HV (>67th pct) -> likely negative GEX -> expect momentum

Study: per session, compute first-30-min and last-30-min returns on IBUS500
and IBDE40. Measure direction match rate and Pearson correlation by HV regime.

If high-HV regime shows significantly higher intraday momentum, this regime
can be used as an entry filter for VWAP Extrema, ORB, and FOMC strategies.

Run from repo root:
    python finance/intraday_pm/backtests/dpm02_gamma_regime.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from finance.utils.intraday import get_bars

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOLS = ["IBUS500", "IBDE40"]
SCHEMA  = "cfd"
START   = datetime(2020, 1, 1, tzinfo=timezone.utc)
END     = datetime(2026, 4, 1, tzinfo=timezone.utc)

TZ_MAP = {
    "IBUS500": ZoneInfo("America/New_York"),
    "IBDE40":  ZoneInfo("Europe/Berlin"),
}
SESSION_OPEN = {
    "IBUS500": (9,  30),   # 09:30 ET
    "IBDE40":  (9,   0),   # 09:00 Frankfurt
}
SESSION_CLOSE = {
    "IBUS500": (16,  0),   # 16:00 ET
    "IBDE40":  (17, 30),   # 17:30 Frankfurt
}
WINDOW_BARS = 6    # 6 x 5-min = 30 min for first/last window
HV_PERIOD   = 20
HV_PCT_WIN  = 252
TRADING_DAYS = 252

RESULTS_APPEND_PATH = "finance/intraday_pm/RESULTS.md"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_5min(symbol: str) -> pd.DataFrame:
    tz = TZ_MAP[symbol]
    df = get_bars(symbol, SCHEMA, START, END, period="5min")
    if df.empty:
        raise RuntimeError(f"No 5-min data for {symbol}")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    df.index = df.index.tz_convert(tz)
    return df[["open", "close"]]


def load_daily_hv(symbol: str) -> pd.Series:
    """Returns daily HV20 percentile series indexed by date."""
    tz = TZ_MAP[symbol]
    df = get_bars(symbol, SCHEMA, START, END, period="1D")
    df = df.rename(columns={"c": "close"})
    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv20 = log_ret.rolling(HV_PERIOD).std() * np.sqrt(TRADING_DAYS)
    hv_pct = hv20.rolling(HV_PCT_WIN, min_periods=HV_PERIOD).rank(pct=True) * 100
    hv_pct.index = pd.to_datetime(hv_pct.index).tz_convert(tz).date
    return hv_pct.dropna()


# ---------------------------------------------------------------------------
# Per-session analysis
# ---------------------------------------------------------------------------
def extract_session_returns(df_5m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    For each session extract:
      - first_ret: return of the first WINDOW_BARS x 5-min bars
      - last_ret:  return of the last WINDOW_BARS x 5-min bars
      - full_ret:  full session return
    """
    open_h, open_m   = SESSION_OPEN[symbol]
    close_h, close_m = SESSION_CLOSE[symbol]

    session_mask = (
        (df_5m.index.hour > open_h) |
        ((df_5m.index.hour == open_h) & (df_5m.index.minute >= open_m))
    ) & (
        (df_5m.index.hour < close_h) |
        ((df_5m.index.hour == close_h) & (df_5m.index.minute <= close_m))
    )
    df = df_5m[session_mask].copy()
    df["date"] = df.index.date

    records = []
    for day, day_df in df.groupby("date"):
        if len(day_df) < WINDOW_BARS * 2 + 2:
            continue
        open_px  = day_df["open"].iloc[0]
        close_px = day_df["close"].iloc[-1]
        if open_px <= 0:
            continue

        first_open  = day_df["open"].iloc[0]
        first_close = day_df["close"].iloc[WINDOW_BARS - 1]
        last_open   = day_df["open"].iloc[-WINDOW_BARS]
        last_close  = day_df["close"].iloc[-1]

        first_ret = (first_close - first_open) / first_open
        last_ret  = (last_close  - last_open)  / last_open
        full_ret  = (close_px    - open_px)    / open_px

        records.append({
            "date"      : day,
            "first_ret" : first_ret,
            "last_ret"  : last_ret,
            "full_ret"  : full_ret,
            "same_dir"  : (first_ret * last_ret) > 0,
        })

    return pd.DataFrame(records).set_index("date")


def classify_regime(sessions: pd.DataFrame, hv_pct: pd.Series) -> pd.DataFrame:
    sessions = sessions.copy()
    sessions["hv_pct"] = sessions.index.map(lambda d: hv_pct.get(d, np.nan))
    sessions = sessions.dropna(subset=["hv_pct"])
    sessions["regime"] = pd.cut(
        sessions["hv_pct"],
        bins=[0, 33, 67, 100],
        labels=["low_hv", "mid_hv", "high_hv"],
    )
    return sessions


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _regime_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n < 5:
        return {}
    dir_match = df["same_dir"].mean() * 100
    corr, pval = pearsonr(df["first_ret"], df["last_ret"])
    avg_first = df["first_ret"].mean() * 100
    avg_last  = df["last_ret"].mean() * 100
    return dict(n=n, dir_match=dir_match, corr=corr, pval=pval,
                avg_first=avg_first, avg_last=avg_last)


def print_report(symbol: str, sessions: pd.DataFrame) -> None:
    sep = "-" * 64
    print(f"\n{sep}")
    print(f"  DPM-02: Gamma Regime Filter  --  {symbol}  {START.date()} to {END.date()}")
    print(f"  Proxy: HV20 pct-rank (252d)  |  30-min windows  |  {len(sessions)} sessions")
    print(sep)
    print(f"  {'Regime':<10}  {'N':>5}  {'Dir match%':>10}  {'Corr':>7}  {'p-val':>7}  {'Interpretation'}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*20}")
    for reg in ["low_hv", "mid_hv", "high_hv"]:
        sub = sessions[sessions["regime"] == reg]
        st = _regime_stats(sub)
        if not st:
            continue
        sig   = "*" if st["pval"] < 0.05 else " "
        interp = "momentum" if st["corr"] > 0.05 else ("mean-rev" if st["corr"] < -0.05 else "neutral")
        print(
            f"  {reg:<10}  {st['n']:>5}  {st['dir_match']:>9.1f}%  "
            f"{st['corr']:>+7.3f}{sig}  {st['pval']:>7.4f}  {interp}"
        )
    print()

    all_st = _regime_stats(sessions)
    if all_st:
        print(f"  Overall direction match: {all_st['dir_match']:.1f}%  "
              f"Corr: {all_st['corr']:+.3f}  (p={all_st['pval']:.4f})")

    # Key finding: does high HV outperform low HV on direction match?
    low  = sessions[sessions["regime"] == "low_hv"]
    high = sessions[sessions["regime"] == "high_hv"]
    if len(low) > 5 and len(high) > 5:
        low_dm  = low["same_dir"].mean() * 100
        high_dm = high["same_dir"].mean() * 100
        delta   = high_dm - low_dm
        print(f"\n  Direction match delta (high_hv - low_hv): {delta:+.1f}pp")
        if delta > 3:
            print("  -> High-HV regime shows stronger intraday momentum.")
            print("     Consider using HV>67th pct as entry filter for VWAP extrema / ORB.")
        elif delta < -3:
            print("  -> Low-HV regime shows stronger momentum (unexpected).")
        else:
            print("  -> No material regime difference. Gamma proxy adds no filter value.")
    print()


def build_section(results: dict[str, pd.DataFrame]) -> str:
    lines = [
        "",
        "---",
        "",
        "## DPM-02: Dealer Gamma Regime Filter (HV20 proxy)",
        "",
        f"Generated: {pd.Timestamp.today().date()}  ",
        f"Period: {START.date()} to {END.date()}  ",
        "Proxy: rolling 252-day HV20 percentile (GEX unavailable)  ",
        "Study: first-30-min return vs last-30-min return by regime  ",
        "Hypothesis: high-HV (negative GEX proxy) -> stronger intraday momentum  ",
        "",
        "| Symbol | Regime | N | Dir match% | Corr | p-val | Signal |",
        "|--------|--------|---|-----------|------|-------|--------|",
    ]

    for symbol, sessions in results.items():
        for reg in ["low_hv", "mid_hv", "high_hv"]:
            sub = sessions[sessions["regime"] == reg]
            st = _regime_stats(sub)
            if not st:
                continue
            sig    = "*" if st["pval"] < 0.05 else ""
            interp = "momentum" if st["corr"] > 0.05 else ("mean-rev" if st["corr"] < -0.05 else "neutral")
            lines.append(
                f"| {symbol} | {reg} | {st['n']} | {st['dir_match']:.1f}% | "
                f"{st['corr']:+.3f}{sig} | {st['pval']:.4f} | {interp} |"
            )

    # Verdict
    any_signal = False
    for symbol, sessions in results.items():
        low  = sessions[sessions["regime"] == "low_hv"]
        high = sessions[sessions["regime"] == "high_hv"]
        if len(low) > 5 and len(high) > 5:
            delta = high["same_dir"].mean() - low["same_dir"].mean()
            if abs(delta) > 0.03:
                any_signal = True

    verdict = "Filter adds value" if any_signal else "No material regime difference"
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
    all_sessions: dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        try:
            df_5m   = load_5min(symbol)
            hv_pct  = load_daily_hv(symbol)
            print(f"  {len(df_5m):,} 5-min bars")

            sessions = extract_session_returns(df_5m, symbol)
            sessions = classify_regime(sessions, hv_pct)
            print(f"  {len(sessions)} complete sessions classified")

            print_report(symbol, sessions)
            all_sessions[symbol] = sessions

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    section = build_section(all_sessions)
    with open(RESULTS_APPEND_PATH, "a", encoding="utf-8") as fh:
        fh.write(section)
    print(f"Appended -> {RESULTS_APPEND_PATH}")


if __name__ == "__main__":
    main()
