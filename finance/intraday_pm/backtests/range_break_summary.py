"""
range_break_summary.py
======================
Aggregates pre-computed following-range-break backtest results across all
instruments and timeframes, then writes finance/intraday_pm/RESULTS.md.

Run from the repo root:
    python finance/intraday_pm/backtests/range_break_summary.py

Input:  N:/My Drive/Trading/Strategies/future_following_range_break/**/*.pkl
        (or cached Parquet files under CACHE_DIR on subsequent runs)
Output: finance/intraday_pm/RESULTS.md
        finance/_data/backtest_results/intraday/range_break/{symbol}_{tf}.parquet  (cache)

On first run the loader reads all daily pkl files from Google Drive and writes
a consolidated Parquet per symbol×timeframe. Every subsequent run loads the
Parquet directly and completes in seconds.

Pass --rebuild to force a fresh read from pkl files.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "N:/My Drive/Trading/Strategies/future_following_range_break"
CACHE_DIR = "finance/_data/backtest_results/intraday/range_break"
RESULTS_PATH = "finance/intraday_pm/RESULTS.md"

SYMBOLS = ["IBDE40", "IBGB100", "IBES35", "IBJP225", "IBUS30", "IBUS500", "IBUST100"]
TIMEFRAMES = ["2m", "5m", "10m"]

# Approximate round-trip spread cost in index points per instrument.
# Used only for the cost note in RESULTS.md — raw move figures are pre-cost.
SPREAD_PTS: dict[str, float] = {
    "IBDE40":  5.0,
    "IBGB100": 2.0,
    "IBES35":  2.0,
    "IBJP225": 5.0,
    "IBUS30":  4.0,
    "IBUS500": 1.0,
    "IBUST100": 2.0,
}

STRATEGY_ORDER = ["cbc", "cbc_10_pct", "cbc_20_pct", "cbc_10_pct_up", "cbc_20_pct_up", "01_pct", "02_pct"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def _load_from_source(symbol: str, tf: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(f"{DATA_DIR}/{symbol}/{symbol}_{tf}_*_follow.parquet"))
    if not files:
        return None
    print(f"  {symbol}/{tf}: reading {len(files)} parquet files...", flush=True)
    parts = [pd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["symbol"] = symbol
    df["timeframe"] = tf
    return df


def load_all(rebuild: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_parts: list[pd.DataFrame] = []

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            cache_path = f"{CACHE_DIR}/{symbol}_{tf}.parquet"

            if not rebuild and os.path.exists(cache_path):
                print(f"  {symbol}/{tf}: loading from cache…", flush=True)
                df = pd.read_parquet(cache_path)
            else:
                df = _load_from_source(symbol, tf)
                if df is None:
                    print(f"  {symbol}/{tf}: no data found, skipping.", flush=True)
                    continue
                df.to_parquet(cache_path, index=False)
                print(f"  {symbol}/{tf}: cached -> {cache_path}", flush=True)

            all_parts.append(df)

    if not all_parts:
        raise RuntimeError(f"No data found. Check DATA_DIR: {DATA_DIR}")
    return pd.concat(all_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_moves(df: pd.DataFrame) -> pd.DataFrame:
    long = df["type"] == "long"

    # Actual trade return (positive = gain)
    df["move"] = np.where(
        long,
        (df["stopout"] - df["entry"]) / df["entry"] * 100,
        (df["entry"] - df["stopout"]) / df["stopout"] * 100,
    )

    # Maximum favourable excursion: for wins = high/low reached; for losses = same as move
    win_long = (~df["loss"]) & long
    win_short = (~df["loss"]) & ~long

    df["move_max"] = df["move"].copy()
    df.loc[win_long, "move_max"] = (
        (df.loc[win_long, "high"] - df.loc[win_long, "entry"])
        / df.loc[win_long, "entry"] * 100
    )
    df.loc[win_short, "move_max"] = (
        (df.loc[win_short, "entry"] - df.loc[win_short, "low"])
        / df.loc[win_short, "low"] * 100
    )
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["symbol", "timeframe", "strategy", "type"])
    agg = grp.agg(
        n=("loss", "count"),
        loss_pct=("loss", lambda s: s.mean() * 100),
        move_mean=("move", "mean"),
        move_median=("move", "median"),
        move_max_mean=("move_max", "mean"),
    ).reset_index()

    # Expectancy proxy: (win_rate * avg_win) + (loss_rate * avg_loss)
    # Derived from the grouped data:
    win_avg = df[~df["loss"]].groupby(["symbol", "timeframe", "strategy", "type"])["move"].mean().rename("avg_win")
    loss_avg = df[df["loss"]].groupby(["symbol", "timeframe", "strategy", "type"])["move"].mean().rename("avg_loss")
    agg = agg.join(win_avg, on=["symbol", "timeframe", "strategy", "type"])
    agg = agg.join(loss_avg, on=["symbol", "timeframe", "strategy", "type"])

    win_rate = (100 - agg["loss_pct"]) / 100
    loss_rate = agg["loss_pct"] / 100
    agg["expectancy"] = (win_rate * agg["avg_win"].fillna(0)) + (loss_rate * agg["avg_loss"].fillna(0))

    # Strategy ordering
    agg["strategy"] = pd.Categorical(agg["strategy"], categories=STRATEGY_ORDER, ordered=True)
    tf_order = pd.Categorical(agg["timeframe"], categories=TIMEFRAMES, ordered=True)
    agg["timeframe"] = tf_order
    return agg.sort_values(["symbol", "timeframe", "strategy", "type"]).reset_index(drop=True)


def best_variant(agg: pd.DataFrame) -> pd.DataFrame:
    """Return the row with highest expectancy per symbol × timeframe × direction."""
    idx = agg.groupby(["symbol", "timeframe", "type"])["expectancy"].idxmax()
    return agg.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------
def fmt(val: float, decimals: int = 2) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:+.{decimals}f}"


def build_markdown(agg: pd.DataFrame, best: pd.DataFrame, df_raw: pd.DataFrame) -> str:
    date_range = (
        df_raw["start"].min().strftime("%Y-%m-%d"),
        df_raw["start"].max().strftime("%Y-%m-%d"),
    )

    lines: list[str] = [
        "# Following Range Break — Backtest Results",
        "",
        f"Generated: {date.today().isoformat()}  ",
        f"Data period: {date_range[0]} -> {date_range[1]}  ",
        "Strategies: `cbc`, `cbc_10_pct`, `cbc_20_pct`, `01_pct`, `02_pct`  ",
        "Timeframes: 2m, 5m, 10m  ",
        "Symbols: IBDE40, IBGB100, IBES35, IBJP225, IBUS30, IBUS500, IBUST100  ",
        "",
        "All move figures are **pre-cost percentages** relative to entry.",
        "See spread cost note at the bottom for net-cost adjustment.",
        "",
        "---",
        "",
        "## Full Results by Symbol",
        "",
    ]

    for symbol in SYMBOLS:
        sym_data = agg[agg["symbol"] == symbol]
        if sym_data.empty:
            continue
        lines.append(f"### {symbol}")
        lines.append("")

        for tf in TIMEFRAMES:
            tf_data = sym_data[sym_data["timeframe"] == tf]
            if tf_data.empty:
                continue
            lines.append(f"**{tf} bars**")
            lines.append("")
            lines.append(
                "| Strategy | Dir | N | Loss% | Exp% | Move mean% | Move median% | MFE mean% |"
            )
            lines.append(
                "|----------|-----|---|-------|------|------------|--------------|-----------|"
            )
            for _, row in tf_data.iterrows():
                lines.append(
                    f"| {row['strategy']} | {row['type']} "
                    f"| {int(row['n'])} "
                    f"| {row['loss_pct']:.1f}% "
                    f"| {fmt(row['expectancy'])}% "
                    f"| {fmt(row['move_mean'])}% "
                    f"| {fmt(row['move_median'])}% "
                    f"| {fmt(row['move_max_mean'])}% |"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "## Optimal Variant per Symbol",
        "",
        "Best expectancy per instrument × timeframe × direction (pre-cost).",
        "",
        "| Symbol | TF | Dir | Strategy | N | Loss% | Expectancy% |",
        "|--------|----|-----|----------|---|-------|-------------|",
    ]
    for _, row in best.sort_values(["symbol", "timeframe", "type"]).iterrows():
        lines.append(
            f"| {row['symbol']} | {row['timeframe']} | {row['type']} "
            f"| {row['strategy']} | {int(row['n'])} "
            f"| {row['loss_pct']:.1f}% | {fmt(row['expectancy'])}% |"
        )

    lines += [
        "",
        "---",
        "",
        "## Cost Note",
        "",
        "Approximate round-trip spread cost per instrument (1 contract):",
        "",
        "| Symbol | Spread (pts) | Entry proxy | Cost% |",
        "|--------|-------------|-------------|-------|",
    ]
    entry_proxies = df_raw.groupby("symbol")["entry"].median()
    for sym, spread in SPREAD_PTS.items():
        entry = entry_proxies.get(sym, float("nan"))
        cost_pct = spread / entry * 100 if not np.isnan(entry) else float("nan")
        lines.append(
            f"| {sym} | {spread:.1f} | {entry:.0f} | {cost_pct:.3f}% |"
        )

    lines += [
        "",
        "To derive net-cost expectancy: subtract the cost% above from the Expectancy% column.",
        "",
        "---",
        "",
        "## Verdict",
        "",
        "*(Populate after reviewing the table above.)*",
        "",
        "| Symbol | TF | Best variant | Net EV | Go / No-go |",
        "|--------|----|-|------|------------|------------|",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force re-read from pkl files")
    args = parser.parse_args()

    print("Loading data…")
    df = load_all(rebuild=args.rebuild)
    print(f"  Loaded {len(df):,} trade rows across {df['symbol'].nunique()} symbols.")

    df = add_moves(df)
    agg = summarise(df)
    best = best_variant(agg)

    md = build_markdown(agg, best, df)

    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"Written -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
