"""
finance.swing_pm.backtests.pead_drift
=====================================
BT-2-S4: PEAD Drift Window Validation

Validates Post-Earnings Announcement Drift for both long (beat) and short (miss)
directions using the momentum_earnings dataset.

Measures forward returns at days 1, 5, 10, 20, 40, 60 post-earnings,
segmented by:
  - SUE quartile (Q1=worst miss, Q4=best beat)
  - Surprise direction (beat vs miss)
  - Market cap class
  - SPY context (supporting vs non-supporting)

Output: markdown reports + parquet trade logs to _data/backtest_results/swing/

Usage
-----
    uv run python -m finance.swing_pm.backtests.pead_drift
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


def _load_earnings_data() -> pd.DataFrame:
    """Load the full dataset and filter to earnings events with valid SUE."""
    print("Loading momentum_earnings dataset...")
    df = load_and_prep_data(YEARS)
    if df.empty:
        print("No data loaded. Ensure the dataset has been generated.")
        return pd.DataFrame()

    # Filter to earnings events only
    mask = df["is_earnings"].fillna(False).astype(bool)
    df_earn = df.loc[mask].copy()

    print(f"Total events: {len(df):,} | Earnings events: {len(df_earn):,}")

    # Compute SUE if not already present
    if "sue" not in df_earn.columns or df_earn["sue"].isna().all():
        if "eps" in df_earn.columns and "eps_est" in df_earn.columns:
            est_abs = df_earn["eps_est"].abs()
            # Guard against division by zero — set SUE to NaN where estimate is zero
            df_earn["sue"] = np.where(
                est_abs > 0,
                (df_earn["eps"] - df_earn["eps_est"]) / est_abs,
                np.nan,
            )
        else:
            print("WARNING: No eps/eps_est columns — SUE cannot be computed.")

    # Compute surprise_dir if not present
    if "surprise_dir" not in df_earn.columns or df_earn["surprise_dir"].isna().all():
        sue = df_earn.get("sue", pd.Series(dtype=float))
        df_earn["surprise_dir"] = np.where(
            sue.isna(), "unknown",
            np.where(sue > 0, "beat",
                     np.where(sue < 0, "miss", "inline")))

    return df_earn


def _add_sue_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add SUE quartile column (Q1=worst miss, Q4=best beat)."""
    valid = df["sue"].notna()
    df = df.copy()
    df["sue_quartile"] = np.nan
    if valid.sum() >= 4:
        df.loc[valid, "sue_quartile"] = pd.qcut(
            df.loc[valid, "sue"], 4,
            labels=["Q1_miss", "Q2", "Q3", "Q4_beat"],
        ).astype(str)
    return df


def run() -> None:
    """Run the full PEAD drift validation and generate reports."""
    df_earn = _load_earnings_data()
    if df_earn.empty:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Overall PEAD drift (all earnings events) ---
    print("\n--- Overall PEAD (all earnings) ---")
    report_all = generate_report(
        df=df_earn,
        label="PEAD — All Earnings Events",
        horizons=HORIZONS,
        segment_cols=["surprise_dir", "market_cap_class", "spy_class"],
    )
    report_all.save(f"{OUTPUT_DIR}/pead_all.md")
    print(report_all.markdown[:500])

    # --- 2. Beats only (long PEAD — PM-02) ---
    df_beats = df_earn[df_earn["surprise_dir"] == "beat"]
    if not df_beats.empty:
        print(f"\n--- Long PEAD: Beats (N={len(df_beats):,}) ---")
        report_beats = generate_report(
            df=df_beats,
            label="PEAD — Beats Only (Long Signal)",
            horizons=HORIZONS,
            segment_cols=["market_cap_class", "spy_class"],
        )
        report_beats.save(f"{OUTPUT_DIR}/pead_beats.md")

    # --- 3. Misses only (short PEAD — BT-3-S1) ---
    df_misses = df_earn[df_earn["surprise_dir"] == "miss"]
    if not df_misses.empty:
        print(f"\n--- Short PEAD: Misses (N={len(df_misses):,}) ---")
        report_misses = generate_report(
            df=df_misses,
            label="PEAD — Misses Only (Short Signal)",
            horizons=HORIZONS,
            segment_cols=["market_cap_class", "spy_class"],
        )
        report_misses.save(f"{OUTPUT_DIR}/pead_misses.md")

    # --- 4. SUE quartile analysis ---
    df_q = _add_sue_quartiles(df_earn)
    valid_quartiles = df_q["sue_quartile"].dropna().unique()

    for q_label in sorted(valid_quartiles):
        df_q_sub = df_q[df_q["sue_quartile"] == q_label]
        if df_q_sub.empty:
            continue
        print(f"\n--- SUE {q_label} (N={len(df_q_sub):,}) ---")
        report_q = generate_report(
            df=df_q_sub,
            label=f"PEAD — SUE {q_label}",
            horizons=HORIZONS,
            segment_cols=["market_cap_class"],
        )
        report_q.save(f"{OUTPUT_DIR}/pead_{q_label.lower()}.md")

    # --- 5. Strong long PEAD: beat + close in top 25% of range ---
    if "close_in_range" in df_earn.columns:
        strong_long_mask = df_beats["close_in_range"] >= 0.75
        if "gappct" in df_beats.columns:
            strong_long_mask = strong_long_mask & (df_beats["gappct"] >= 10)
        df_strong_long = df_beats[strong_long_mask]
        if not df_strong_long.empty:
            print(f"\n--- Strong Long PEAD: Beat + Top 25% Close + Gap ≥10% (N={len(df_strong_long):,}) ---")
            report_strong = generate_report(
                df=df_strong_long,
                label="PEAD — Strong Long (Beat + Top25% + Gap≥10%)",
                horizons=HORIZONS,
                segment_cols=["market_cap_class", "spy_class"],
            )
            report_strong.save(f"{OUTPUT_DIR}/pead_strong_long.md")

    # --- 6. Strong short PEAD: miss + close in bottom 25% of range ---
    if "close_in_range" in df_earn.columns:
        strong_short_mask = df_misses["close_in_range"] <= 0.25
        if "gappct" in df_misses.columns:
            strong_short_mask = strong_short_mask & (df_misses["gappct"] <= -5)
        df_strong_short = df_misses[strong_short_mask]
        if not df_strong_short.empty:
            print(f"\n--- Strong Short PEAD: Miss + Bottom 25% Close + Gap ≤-5% (N={len(df_strong_short):,}) ---")
            report_short = generate_report(
                df=df_strong_short,
                label="PEAD — Strong Short (Miss + Bottom25% + Gap≤-5%)",
                horizons=HORIZONS,
                segment_cols=["market_cap_class", "spy_class"],
            )
            report_short.save(f"{OUTPUT_DIR}/pead_strong_short.md")

    # --- 7. Save the filtered earnings data as parquet for further analysis ---
    df_earn.to_parquet(f"{OUTPUT_DIR}/pead_earnings_events.parquet", index=False)
    print(f"\nSaved {len(df_earn):,} earnings events to {OUTPUT_DIR}/pead_earnings_events.parquet")
    print("Done.")


if __name__ == "__main__":
    run()
