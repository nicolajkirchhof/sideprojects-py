"""
finance.swing_pm.backtests.consecutive_miss
============================================
BT-3-S2: Consecutive Miss + Guidance Cut

Analyses forward returns after consecutive earnings misses (2+ sequential miss
events per symbol). Tests whether repeated misses predict stronger negative
drift than single misses.

Uses the momentum_earnings dataset which already contains consecutive_beats
and surprise_dir per earnings event.

Output: markdown reports to _data/backtest_results/swing/

Usage
-----
    uv run python -m finance.swing_pm.backtests.consecutive_miss
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from finance.utils.momentum_data import load_and_prep_data
from finance.swing_pm.backtests.backtest_report import generate_report

HORIZONS = [1, 5, 10, 20, 40, 60]
YEARS = range(2016, 2027)
OUTPUT_DIR = "finance/_data/backtest_results/swing"


def _identify_consecutive_misses(df_earn: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'consecutive_misses' column: running count of sequential miss events
    per symbol. Resets to 0 on any non-miss event.

    The dataset has consecutive_beats (resets on miss). We need the inverse:
    consecutive misses that reset on beat.
    """
    df = df_earn.sort_values(["symbol", "date"]).copy() if "symbol" in df_earn.columns else df_earn.sort_values("date").copy()

    running = 0
    prev_symbol = None
    miss_counts = []

    for _, row in df.iterrows():
        sym = row.get("symbol", "")
        if sym != prev_symbol:
            running = 0
            prev_symbol = sym

        sdir = row.get("surprise_dir", "unknown")
        if sdir == "miss":
            running += 1
        else:
            running = 0

        miss_counts.append(running)

    df["consecutive_misses"] = miss_counts
    return df


def run() -> None:
    """Run the consecutive miss analysis."""
    print("Loading momentum_earnings dataset...")
    df = load_and_prep_data(YEARS)
    if df.empty:
        print("No data loaded.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Filter to earnings events
    df_earn = df[df["is_earnings"].fillna(False).astype(bool)].copy()
    print(f"Total events: {len(df):,} | Earnings events: {len(df_earn):,}")

    # Need symbol column for grouping
    if "symbol" not in df_earn.columns:
        # Try to reconstruct from the dataset — the momentum_earnings dataset
        # might not include symbol in load_and_prep_data. Check.
        print("WARNING: 'symbol' column not found. Cannot compute consecutive misses.")
        return

    # Compute surprise_dir if needed
    if "surprise_dir" not in df_earn.columns or df_earn["surprise_dir"].isna().all():
        if "sue" in df_earn.columns:
            sue = df_earn["sue"]
            df_earn["surprise_dir"] = np.where(
                sue.isna(), "unknown",
                np.where(sue > 0, "beat",
                         np.where(sue < 0, "miss", "inline")))

    df_earn = _identify_consecutive_misses(df_earn)

    # --- 1. Single miss (baseline) ---
    df_single = df_earn[df_earn["consecutive_misses"] == 1]
    print(f"\n--- Single Miss (1st miss, N={len(df_single):,}) ---")
    if not df_single.empty:
        report_single = generate_report(
            df=df_single,
            label="Single Miss (1st consecutive miss)",
            horizons=HORIZONS,
            segment_cols=["market_cap_class", "spy_class"],
        )
        report_single.save(f"{OUTPUT_DIR}/consecutive_miss_single.md")

    # --- 2. Double miss (2nd consecutive) ---
    df_double = df_earn[df_earn["consecutive_misses"] == 2]
    print(f"\n--- Double Miss (2nd consecutive, N={len(df_double):,}) ---")
    if not df_double.empty:
        report_double = generate_report(
            df=df_double,
            label="Double Miss (2nd consecutive miss)",
            horizons=HORIZONS,
            segment_cols=["market_cap_class", "spy_class"],
        )
        report_double.save(f"{OUTPUT_DIR}/consecutive_miss_double.md")

    # --- 3. Triple+ miss (3rd or more consecutive) ---
    df_triple = df_earn[df_earn["consecutive_misses"] >= 3]
    print(f"\n--- Triple+ Miss (3rd+ consecutive, N={len(df_triple):,}) ---")
    if not df_triple.empty:
        report_triple = generate_report(
            df=df_triple,
            label="Triple+ Miss (3rd+ consecutive miss)",
            horizons=HORIZONS,
            segment_cols=["market_cap_class", "spy_class"],
        )
        report_triple.save(f"{OUTPUT_DIR}/consecutive_miss_triple.md")

    # --- 4. Any consecutive miss (2+) vs single miss comparison ---
    df_consec = df_earn[df_earn["consecutive_misses"] >= 2]
    print(f"\n--- All Consecutive Misses (2+, N={len(df_consec):,}) ---")
    if not df_consec.empty:
        report_consec = generate_report(
            df=df_consec,
            label="All Consecutive Misses (2+ sequential)",
            horizons=HORIZONS,
            segment_cols=["market_cap_class", "spy_class"],
        )
        report_consec.save(f"{OUTPUT_DIR}/consecutive_miss_all.md")

    print(f"\nReports saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    run()
