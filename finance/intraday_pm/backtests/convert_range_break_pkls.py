"""
convert_range_break_pkls.py
============================
One-time migration: reads all daily *_follow.pkl files from Google Drive
and writes a consolidated *_follow.parquet per symbol × timeframe to the
local backtest results directory.

Run from the repo root:
    python finance/intraday_pm/backtests/convert_range_break_pkls.py

After this script completes successfully the pkl files on Google Drive are
no longer needed by any active code.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime

import pandas as pd

PKL_DIR = "N:/My Drive/Trading/Strategies/future_following_range_break"
OUT_DIR = "finance/_data/backtest_results/intraday/range_break"

SYMBOLS = ["IBDE40", "IBGB100", "IBES35", "IBJP225", "IBUS30", "IBUS500", "IBUST100"]
TIMEFRAMES = ["2m", "5m", "10m"]


def convert(symbol: str, tf: str) -> bool:
    out_path = f"{OUT_DIR}/{symbol}_{tf}.parquet"
    if os.path.exists(out_path):
        print(f"  {symbol}/{tf}: already exists, skipping.")
        return True

    files = sorted(glob.glob(f"{PKL_DIR}/{symbol}/{symbol}_{tf}_*_follow.pkl"))
    if not files:
        print(f"  {symbol}/{tf}: no pkl files found, skipping.")
        return False

    print(f"  {symbol}/{tf}: reading {len(files)} files...", flush=True)
    parts = [pd.read_pickle(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["symbol"] = symbol
    df["timeframe"] = tf
    df.to_parquet(out_path, index=False)
    print(f"  {symbol}/{tf}: wrote {len(df):,} rows -> {out_path}")
    return True


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = datetime.now()
    ok = 0
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            if convert(symbol, tf):
                ok += 1
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nDone: {ok}/{len(SYMBOLS) * len(TIMEFRAMES)} converted in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
