"""
finance.swing_pm.backtests.atrp_breakout
========================================
ATRP Breakout Forward Return Analysis

Analyses forward returns after ATRP breakout events (|daily move| > 1.5× ATR20),
segmented by:
  - Event move magnitude and direction (signed xATR)
  - Market cap class
  - SPY context (supporting vs non-supporting)
  - Episodic Pivot overlap (events that also qualify as EP)

Also analyses EP-specific events and EMA reclaim events when available.

Output: markdown reports + parquet to _data/backtest_results/swing/

Usage
-----
    uv run python -m finance.swing_pm.backtests.atrp_breakout
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


def _load_event_data() -> pd.DataFrame:
    """Load full dataset for event analysis."""
    print("Loading momentum_earnings dataset...")
    df = load_and_prep_data(YEARS)
    if df.empty:
        print("No data loaded.")
    else:
        print(f"Total events: {len(df):,}")
    return df


def _add_move_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Categorise event_move into signed magnitude buckets."""
    df = df.copy()
    if "event_move" not in df.columns:
        df["move_bucket"] = "unknown"
        return df

    em = df["event_move"]
    conditions = [
        em >= 3.0,
        (em >= 1.5) & (em < 3.0),
        (em >= 0) & (em < 1.5),
        (em >= -1.5) & (em < 0),
        (em >= -3.0) & (em < -1.5),
        em < -3.0,
    ]
    labels = ["+3x+", "+1.5–3x", "+0–1.5x", "-0–1.5x", "-1.5–3x", "-3x+"]
    df["move_bucket"] = np.select(conditions, labels, default="unknown")
    return df


def run() -> None:
    """Run event-based forward return analyses."""
    df = _load_event_data()
    if df.empty:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===================================================================
    # 1. ATRP Breakout Events
    # ===================================================================
    df_atrp = df[df["evt_atrp_breakout"].fillna(False).astype(bool)]
    if not df_atrp.empty:
        df_atrp = _add_move_bucket(df_atrp)
        print(f"\n--- ATRP Breakout (N={len(df_atrp):,}) ---")
        report = generate_report(
            df=df_atrp,
            label="ATRP Breakout (|move| > 1.5× ATR20)",
            horizons=HORIZONS,
            segment_cols=["move_bucket", "market_cap_class", "spy_class"],
        )
        report.save(f"{OUTPUT_DIR}/atrp_breakout_all.md")

        # Split by direction
        df_atrp_long = df_atrp[df_atrp["event_move"] > 0]
        df_atrp_short = df_atrp[df_atrp["event_move"] < 0]

        if not df_atrp_long.empty:
            report_long = generate_report(
                df=df_atrp_long,
                label="ATRP Breakout — Bullish (positive move)",
                horizons=HORIZONS,
                segment_cols=["move_bucket", "market_cap_class", "spy_class"],
            )
            report_long.save(f"{OUTPUT_DIR}/atrp_breakout_long.md")

        if not df_atrp_short.empty:
            report_short = generate_report(
                df=df_atrp_short,
                label="ATRP Breakout — Bearish (negative move)",
                horizons=HORIZONS,
                segment_cols=["move_bucket", "market_cap_class", "spy_class"],
            )
            report_short.save(f"{OUTPUT_DIR}/atrp_breakout_short.md")

    # ===================================================================
    # 2. Episodic Pivot Events (Type A)
    # ===================================================================
    if "evt_episodic_pivot" in df.columns:
        df_ep = df[df["evt_episodic_pivot"].fillna(False).astype(bool)]
        if not df_ep.empty:
            df_ep = _add_move_bucket(df_ep)
            print(f"\n--- Episodic Pivot (N={len(df_ep):,}) ---")
            report_ep = generate_report(
                df=df_ep,
                label="Episodic Pivot (gap≥10%, RVOL≥5×, close in range quartile)",
                horizons=HORIZONS,
                segment_cols=["move_bucket", "market_cap_class", "spy_class"],
            )
            report_ep.save(f"{OUTPUT_DIR}/episodic_pivot.md")

    # ===================================================================
    # 3. EMA Reclaim Events (Type C)
    # ===================================================================
    if "evt_ema_reclaim" in df.columns:
        df_ema = df[df["evt_ema_reclaim"].fillna(False).astype(bool)]
        if not df_ema.empty:
            print(f"\n--- EMA Reclaim (N={len(df_ema):,}) ---")
            report_ema = generate_report(
                df=df_ema,
                label="EMA Reclaim (pullback to 10/20 MA, stack intact, low vol)",
                horizons=HORIZONS,
                segment_cols=["market_cap_class", "spy_class"],
            )
            report_ema.save(f"{OUTPUT_DIR}/ema_reclaim.md")

    # ===================================================================
    # 4. BB Lower Touch Events (PM-09 Mean Reversion)
    # ===================================================================
    df_bb = df[df["evt_bb_lower_touch"].fillna(False).astype(bool)]
    if not df_bb.empty:
        print(f"\n--- BB Lower Touch (N={len(df_bb):,}) ---")
        report_bb = generate_report(
            df=df_bb,
            label="BB Lower Touch (mean reversion to trend, PM-09)",
            horizons=HORIZONS,
            segment_cols=["market_cap_class", "spy_class"],
        )
        report_bb.save(f"{OUTPUT_DIR}/bb_lower_touch.md")

    # ===================================================================
    # 5. Selloff Events (PM-08 Overnight Reversal candidates)
    # ===================================================================
    if "evt_selloff" in df.columns:
        df_sell = df[df["evt_selloff"].fillna(False).astype(bool)]
        if not df_sell.empty:
            print(f"\n--- Selloff (N={len(df_sell):,}) ---")
            report_sell = generate_report(
                df=df_sell,
                label="Selloff (pct<-2%, RVOL>1.5, Stage 2 — overnight reversal candidates)",
                horizons=HORIZONS,
                segment_cols=["market_cap_class", "spy_class"],
            )
            report_sell.save(f"{OUTPUT_DIR}/selloff.md")

    # ===================================================================
    # 6. Green Line Breakout Events
    # ===================================================================
    df_gl = df[df["evt_green_line_breakout"].fillna(False).astype(bool)]
    if not df_gl.empty:
        print(f"\n--- Green Line Breakout (N={len(df_gl):,}) ---")
        report_gl = generate_report(
            df=df_gl,
            label="Green Line Breakout (ATH after consolidation)",
            horizons=HORIZONS,
            segment_cols=["market_cap_class", "spy_class"],
        )
        report_gl.save(f"{OUTPUT_DIR}/green_line_breakout.md")

    # ===================================================================
    # 7. Pre-Earnings Events (PM-03)
    # ===================================================================
    if "evt_pre_earnings" in df.columns:
        df_pre = df[df["evt_pre_earnings"].fillna(False).astype(bool)]
        if not df_pre.empty:
            print(f"\n--- Pre-Earnings (N={len(df_pre):,}) ---")
            # For pre-earnings, the key horizon is T-14 to T-1 (≈10 trading days)
            report_pre = generate_report(
                df=df_pre,
                label="Pre-Earnings Anticipation (T-14, Stage 2, ≥3 consecutive beats)",
                horizons=[5, 10, 14, 20],
                segment_cols=["market_cap_class", "spy_class"],
            )
            report_pre.save(f"{OUTPUT_DIR}/pre_earnings.md")

    print(f"\nAll reports saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    run()
