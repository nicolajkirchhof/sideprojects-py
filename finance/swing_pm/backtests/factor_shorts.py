"""
finance.swing_pm.backtests.factor_shorts
=========================================
BT-3-S3: Accruals Factor Short Screen
BT-3-S4: Piotroski F-Score Long/Short Filter

Queries Dolt financial statements quarterly, computes accruals ratio and
F-Score per symbol per quarter, joins with EOD OHLCV for forward returns.
Outputs decile-level return tables.

Usage
-----
    uv run python -m finance.swing_pm.backtests.factor_shorts
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sqlalchemy import text

from finance.utils.dolt_data import db_earnings_connection, db_stocks_connection
from finance.utils.fundamentals import compute_accruals, compute_fscore

OUTPUT_DIR = "finance/_data/backtest_results/swing"
FORWARD_DAYS = [21, 63, 126, 252]  # ~1 quarter, 3 months, 6 months, 1 year


# ---------------------------------------------------------------------------
# Data loading from Dolt
# ---------------------------------------------------------------------------

def _load_quarterly_financials() -> pd.DataFrame:
    """
    Load quarterly financial data from Dolt: income_statement, cash_flow_statement,
    balance_sheet_assets, balance_sheet_liabilities. Join all on (act_symbol, date).

    Returns one row per symbol per quarter with all fields needed for
    accruals and F-Score computation.
    """
    print("Loading quarterly financials from Dolt...")
    t0 = time.time()

    # Load each table
    df_cf = pd.read_sql(text("""
        SELECT act_symbol, date,
               net_income, net_cash_from_operating_activities
        FROM cash_flow_statement
        WHERE period = 'Quarter'
        ORDER BY act_symbol, date
    """), db_earnings_connection)

    df_bs_a = pd.read_sql(text("""
        SELECT act_symbol, date,
               total_assets, total_current_assets
        FROM balance_sheet_assets
        WHERE period = 'Quarter'
        ORDER BY act_symbol, date
    """), db_earnings_connection)

    df_bs_l = pd.read_sql(text("""
        SELECT act_symbol, date,
               long_term_debt, total_current_liabilities
        FROM balance_sheet_liabilities
        WHERE period = 'Quarter'
        ORDER BY act_symbol, date
    """), db_earnings_connection)

    df_is = pd.read_sql(text("""
        SELECT act_symbol, date,
               sales, cost_of_goods, gross_profit, net_income as is_net_income,
               average_shares
        FROM income_statement
        WHERE period = 'Quarter'
        ORDER BY act_symbol, date
    """), db_earnings_connection)

    print(f"  Cash flow: {len(df_cf):,} rows")
    print(f"  Balance sheet assets: {len(df_bs_a):,} rows")
    print(f"  Balance sheet liabilities: {len(df_bs_l):,} rows")
    print(f"  Income statement: {len(df_is):,} rows")

    # Convert dates
    for df in [df_cf, df_bs_a, df_bs_l, df_is]:
        df["date"] = pd.to_datetime(df["date"])

    # Join all on (act_symbol, date)
    df = df_cf.merge(df_bs_a, on=["act_symbol", "date"], how="outer")
    df = df.merge(df_bs_l, on=["act_symbol", "date"], how="outer")
    df = df.merge(df_is, on=["act_symbol", "date"], how="outer")

    # Use income statement net_income if cash flow doesn't have it
    if "net_income" in df.columns and "is_net_income" in df.columns:
        df["net_income"] = df["net_income"].fillna(df["is_net_income"])
    df = df.drop(columns=["is_net_income"], errors="ignore")

    # Convert decimal columns to float
    numeric_cols = [c for c in df.columns if c not in ("act_symbol", "date")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["act_symbol", "date"]).reset_index(drop=True)

    elapsed = time.time() - t0
    print(f"  Merged: {len(df):,} rows, {df['act_symbol'].nunique():,} symbols in {elapsed:.1f}s")

    return df


def _add_prior_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, add prior quarter values as prev_* columns.
    Groups by act_symbol, shifts by 1 within each group.
    """
    df = df.sort_values(["act_symbol", "date"]).copy()

    # Columns to shift
    shift_cols = [
        "net_income", "total_assets", "net_cash_from_operating_activities",
        "long_term_debt", "total_current_assets", "total_current_liabilities",
        "gross_profit", "sales", "average_shares",
    ]

    existing = [c for c in shift_cols if c in df.columns]
    shifted = df.groupby("act_symbol")[existing].shift(1)
    shifted.columns = [f"prev_{c}" for c in existing]

    return pd.concat([df, shifted], axis=1)


def _load_quarterly_returns(symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Load close prices from Dolt ohlcv for forward return computation.
    Loads in batches of 500 symbols to avoid query size limits.
    """
    print("Loading EOD prices for forward returns...")
    t0 = time.time()

    if symbols is None:
        # Fallback: load everything (slow)
        df = pd.read_sql(text("""
            SELECT act_symbol, date, close
            FROM ohlcv
            WHERE date >= '2015-01-01'
            ORDER BY act_symbol, date
        """), db_stocks_connection)
    else:
        # Batch load by symbol chunks
        batch_size = 500
        dfs = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            placeholders = ",".join(f"'{s}'" for s in batch)
            query = text(f"""
                SELECT act_symbol, date, close
                FROM ohlcv
                WHERE act_symbol IN ({placeholders})
                  AND date >= '2015-01-01'
                ORDER BY act_symbol, date
            """)
            dfs.append(pd.read_sql(query, db_stocks_connection))
            print(f"  Batch {i // batch_size + 1}: {len(batch)} symbols loaded")
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    elapsed = time.time() - t0
    print(f"  Loaded {len(df):,} price rows in {elapsed:.1f}s")

    return df


def _compute_forward_returns(
    df_signals: pd.DataFrame,
    df_prices: pd.DataFrame,
    forward_days: list[int],
) -> pd.DataFrame:
    """
    For each signal row (act_symbol, date), compute forward returns at given horizons.

    Uses merge_asof to find the nearest trading day, then looks forward.
    """
    print("Computing forward returns...")
    results = df_signals[["act_symbol", "date"]].copy()

    for fwd in forward_days:
        col = f"fwd_{fwd}d"
        returns = []

        for _, row in df_signals.iterrows():
            sym = row["act_symbol"]
            sig_date = row["date"]

            sym_prices = df_prices[df_prices["act_symbol"] == sym].set_index("date")["close"]
            if sym_prices.empty:
                returns.append(np.nan)
                continue

            # Find nearest trading day on or after signal date
            valid_dates = sym_prices.index[sym_prices.index >= sig_date]
            if len(valid_dates) == 0:
                returns.append(np.nan)
                continue

            entry_date = valid_dates[0]
            entry_price = sym_prices[entry_date]

            # Find price at forward horizon
            target_date = entry_date + pd.Timedelta(days=fwd)
            future_dates = sym_prices.index[(sym_prices.index >= target_date)]
            if len(future_dates) == 0:
                returns.append(np.nan)
                continue

            exit_price = sym_prices[future_dates[0]]
            returns.append((exit_price / entry_price - 1) * 100)

        results[col] = returns

    return results


def _compute_forward_returns_vectorized(
    df_signals: pd.DataFrame,
    df_prices: pd.DataFrame,
    forward_days: list[int],
) -> pd.DataFrame:
    """
    Vectorized forward return computation using merge_asof.
    Much faster than the per-row approach for large datasets.
    """
    print("Computing forward returns (vectorized)...")
    t0 = time.time()

    df_p = df_prices.sort_values(["act_symbol", "date"]).reset_index(drop=True)
    df_s = df_signals[["act_symbol", "date"]].sort_values(["act_symbol", "date"]).reset_index(drop=True)

    # Build price index per symbol for fast lookup
    price_map: dict[str, pd.Series] = {}
    for sym, grp in df_p.groupby("act_symbol"):
        price_map[sym] = grp.set_index("date")["close"]

    results = df_s.copy()
    for fwd in forward_days:
        col = f"fwd_{fwd}d"
        returns = np.full(len(df_s), np.nan)

        for idx, row in df_s.iterrows():
            sym = row["act_symbol"]
            sig_date = row["date"]

            prices = price_map.get(sym)
            if prices is None or prices.empty:
                continue

            # Entry: nearest trading day on or after signal date
            entry_mask = prices.index >= sig_date
            if not entry_mask.any():
                continue
            entry_price = prices[entry_mask].iloc[0]

            # Exit: nearest trading day on or after signal_date + fwd days
            target = sig_date + pd.Timedelta(days=fwd)
            exit_mask = prices.index >= target
            if not exit_mask.any():
                continue
            exit_price = prices[exit_mask].iloc[0]

            returns[idx] = (exit_price / entry_price - 1) * 100

        results[col] = returns

    elapsed = time.time() - t0
    print(f"  Forward returns computed in {elapsed:.1f}s")

    return results


def _format_decile_report(
    df: pd.DataFrame,
    factor_col: str,
    label: str,
    n_quantiles: int = 10,
) -> str:
    """Format a decile return table as markdown."""
    fwd_cols = [c for c in df.columns if c.startswith("fwd_")]
    if not fwd_cols:
        return f"*No forward return columns for {label}*"

    valid = df[factor_col].notna()
    df_valid = df[valid].copy()
    if len(df_valid) < n_quantiles:
        return f"*Insufficient data for {label} (N={len(df_valid)})*"

    df_valid["decile"] = pd.qcut(df_valid[factor_col], n_quantiles, labels=False, duplicates="drop") + 1

    lines = [
        f"# Factor Report: {label}",
        f"",
        f"**Events:** {len(df_valid):,}",
        f"**Factor:** {factor_col}",
        f"**Quantiles:** {n_quantiles}",
        f"",
    ]

    # Summary table
    fwd_headers = " | ".join(c.replace("fwd_", "") for c in fwd_cols)
    fwd_sep = " | ".join("-------" for _ in fwd_cols)

    lines.extend([
        f"| Decile | N | {factor_col} Mean | {fwd_headers} |",
        f"|--------|---|----------|{fwd_sep}|",
    ])

    for d in sorted(df_valid["decile"].unique()):
        group = df_valid[df_valid["decile"] == d]
        n = len(group)
        factor_mean = group[factor_col].mean()
        fwd_vals = " | ".join(f"{group[c].mean():+.2f}" for c in fwd_cols)
        lines.append(f"| D{d:02d} | {n:,} | {factor_mean:+.4f} | {fwd_vals} |")

    # Long-short spread
    d_low = df_valid[df_valid["decile"] == 1]
    d_high = df_valid[df_valid["decile"] == df_valid["decile"].max()]
    if not d_low.empty and not d_high.empty:
        lines.extend([
            f"",
            f"**Long-Short Spread (D1 - D{int(df_valid['decile'].max())}):**",
        ])
        for c in fwd_cols:
            spread = d_low[c].mean() - d_high[c].mean()
            lines.append(f"- {c.replace('fwd_', '')}: {spread:+.2f}%")

    return "\n".join(lines)


def run() -> None:
    """Run factor-based backtests: Accruals + F-Score."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load financial data
    df_fin = _load_quarterly_financials()
    if df_fin.empty:
        print("No financial data loaded.")
        return

    # Filter to 2016+ (matches our PEAD dataset)
    df_fin = df_fin[df_fin["date"] >= "2016-01-01"].copy()
    print(f"Post-2016 financials: {len(df_fin):,} rows, {df_fin['act_symbol'].nunique():,} symbols")

    # Add prior quarter for F-Score deltas
    df_fin = _add_prior_quarter(df_fin)

    # ===================================================================
    # BT-3-S3: Accruals Anomaly
    # ===================================================================
    print("\n=== BT-3-S3: Accruals Anomaly ===")
    df_accruals = compute_accruals(df_fin)
    df_accruals_valid = df_accruals[df_accruals["accruals_ratio"].notna()].copy()
    print(f"Valid accruals ratios: {len(df_accruals_valid):,}")

    # Load prices for forward returns — only for symbols with financial data
    fin_symbols = df_accruals_valid["act_symbol"].unique().tolist()
    df_prices = _load_quarterly_returns(symbols=fin_symbols)
    df_accruals_fwd = _compute_forward_returns_vectorized(
        df_accruals_valid, df_prices, FORWARD_DAYS,
    )

    # Merge factor values back
    df_accruals_report = df_accruals_valid[["act_symbol", "date", "accruals_ratio"]].copy()
    df_accruals_report = df_accruals_report.merge(
        df_accruals_fwd, on=["act_symbol", "date"], how="left",
    )

    accruals_md = _format_decile_report(
        df_accruals_report, "accruals_ratio",
        "Accruals Anomaly (Sloan 1996) — High accruals = short, Low = long",
    )

    with open(f"{OUTPUT_DIR}/accruals_factor.md", "w", encoding="utf-8") as f:
        f.write(accruals_md)
    print(f"Saved accruals report")

    # ===================================================================
    # BT-3-S4: Piotroski F-Score
    # ===================================================================
    print("\n=== BT-3-S4: Piotroski F-Score ===")
    df_fscore = compute_fscore(df_fin)
    valid_fscore = df_fscore["fscore"].notna().sum()
    print(f"Valid F-Scores: {valid_fscore:,}")

    df_fscore_valid = df_fscore[df_fscore["fscore"].notna()].copy()
    df_fscore_fwd = _compute_forward_returns_vectorized(
        df_fscore_valid, df_prices, FORWARD_DAYS,
    )

    df_fscore_report = df_fscore_valid[["act_symbol", "date", "fscore"]].copy()
    df_fscore_report = df_fscore_report.merge(
        df_fscore_fwd, on=["act_symbol", "date"], how="left",
    )

    # F-Score uses score buckets not deciles (0-9 discrete)
    fwd_cols = [c for c in df_fscore_report.columns if c.startswith("fwd_")]
    if fwd_cols:
        fwd_headers = " | ".join(c.replace("fwd_", "") for c in fwd_cols)
        fwd_sep = " | ".join("-------" for _ in fwd_cols)

        lines = [
            "# Factor Report: Piotroski F-Score (2000)",
            "",
            f"**Events:** {len(df_fscore_report):,}",
            f"**Factor:** fscore (0-9)",
            "",
            f"| F-Score | N | {fwd_headers} |",
            f"|---------|---|{fwd_sep}|",
        ]

        for score in range(10):
            group = df_fscore_report[df_fscore_report["fscore"] == score]
            if group.empty:
                continue
            fwd_vals = " | ".join(f"{group[c].mean():+.2f}" for c in fwd_cols)
            lines.append(f"| {score} | {len(group):,} | {fwd_vals} |")

        # Long (>=8) vs Short (<=2)
        long_group = df_fscore_report[df_fscore_report["fscore"] >= 8]
        short_group = df_fscore_report[df_fscore_report["fscore"] <= 2]
        if not long_group.empty and not short_group.empty:
            lines.extend([
                "",
                "**Long (F>=8) vs Short (F<=2) Spread:**",
            ])
            for c in fwd_cols:
                spread = long_group[c].mean() - short_group[c].mean()
                lines.append(f"- {c.replace('fwd_', '')}: {spread:+.2f}%")

        fscore_md = "\n".join(lines)
    else:
        fscore_md = "*No forward return columns computed*"

    with open(f"{OUTPUT_DIR}/fscore_factor.md", "w", encoding="utf-8") as f:
        f.write(fscore_md)
    print(f"Saved F-Score report")

    # Save raw data for further analysis
    df_accruals_report.to_parquet(f"{OUTPUT_DIR}/accruals_factor_data.parquet", index=False)
    df_fscore_report.to_parquet(f"{OUTPUT_DIR}/fscore_factor_data.parquet", index=False)

    print(f"\nAll factor reports saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    run()
