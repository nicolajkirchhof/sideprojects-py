"""
finance.utils.momentum_data
=============================
Data loading and preparation for the momentum/earnings analysis dashboard
and swing trading backtests (PEAD, ATRP, EMA reclaim, etc.).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore

# ---------------------------------------------------------------------------
# Constants — match the data generation script offsets
# ---------------------------------------------------------------------------
DAILY_FORWARD_DAYS = 60
WEEKLY_FORWARD_WEEKS = 12

_DATASET_DIR = "finance/_data/research/swing/momentum_earnings"
_TICKER_DIR = f"{_DATASET_DIR}/ticker"

# ---------------------------------------------------------------------------
# Column registry
# ---------------------------------------------------------------------------

def _required_columns() -> list[str]:
    """
    Return the sorted list of columns that load_and_prep_data will attempt
    to read from the yearly parquet files.

    Exposed at module level so backtests and tests can inspect the schema.
    """
    cols: set[str] = {
        # core
        "date", "original_price", "c0", "cpct0", "atrp200", "is_earnings", "is_etf",
        "spy0", "spy5", "market_cap_class",
        # event types
        "evt_atrp_breakout", "evt_green_line_breakout", "evt_bb_lower_touch",
        "evt_episodic_pivot", "evt_pre_earnings", "evt_ema_reclaim", "evt_selloff",
        # earnings / PEAD columns
        "sue", "surprise_dir", "close_in_range", "consecutive_beats",
        "gappct", "eps", "eps_est",
        # momentum filters
        "1M_chg", "3M_chg", "6M_chg", "12M_chg",
        "ma10_dist0", "ma20_dist0", "ma50_dist0", "ma100_dist0", "ma200_dist0",
        "spy_ma10_dist0", "spy_ma20_dist0", "spy_ma50_dist0", "spy_ma100_dist0", "spy_ma200_dist0",
    }

    # Trajectory / dist / cond filter — daily (1..60) + weekly (1..12)
    for i in range(1, DAILY_FORWARD_DAYS + 1):
        cols.add(f"cpct{i}")
        cols.add(f"ma5_dist{i}")
        cols.add(f"ma10_dist{i}")
        cols.add(f"ma20_dist{i}")
        cols.add(f"ma50_dist{i}")
    for i in range(1, WEEKLY_FORWARD_WEEKS + 1):
        cols.add(f"w_cpct{i}")
        cols.add(f"w_ma5_dist{i}")
        cols.add(f"w_ma10_dist{i}")
        cols.add(f"w_ma20_dist{i}")
        cols.add(f"w_ma50_dist{i}")

    # Distribution-over-time options — daily (1..60) + weekly (1..12)
    dist_metrics = ["ma5_slope", "ma10_slope", "ma20_slope", "ma50_slope",
                    "rvol20", "hv20", "atrp20"]
    for m in dist_metrics:
        for i in range(1, DAILY_FORWARD_DAYS + 1):
            cols.add(f"{m}{i}")
        for i in range(1, WEEKLY_FORWARD_WEEKS + 1):
            cols.add(f"w_{m}{i}")

    return sorted(cols)


def load_ticker_earnings_events(symbol: str) -> pd.DataFrame:
    """
    Load per-ticker earnings events from the momentum_earnings dataset.

    Returns a DataFrame with one row per earnings event and the cpct[-60..60]
    forward/backward return columns. Adds an `eps_surprise` column
    (eps - eps_est) and a `surprise_dir` column ('beat' / 'miss' / 'unknown').
    Returns an empty DataFrame if the ticker parquet is missing.
    """
    path = f"{_TICKER_DIR}/{symbol.upper()}.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if df.empty or 'is_earnings' not in df.columns:
        return pd.DataFrame()

    df = df[df['is_earnings'].fillna(False).astype(bool)].copy()
    if df.empty:
        return df

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    new_cols: dict[str, pd.Series | float] = {}
    eps_surprise = df['eps'] - df['eps_est'] if {'eps', 'eps_est'} <= set(df.columns) else pd.Series(np.nan, index=df.index)
    new_cols['eps_surprise'] = eps_surprise
    new_cols['surprise_dir'] = np.where(
        eps_surprise.isna(), 'unknown',
        np.where(eps_surprise > 0, 'beat',
                 np.where(eps_surprise < 0, 'miss', 'inline'))
    )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df.sort_values('date').reset_index(drop=True)


def load_and_prep_data(years: range) -> pd.DataFrame:
    """
    Loads and standardizes the momentum/earnings dataset for the dashboard
    and backtests.
    """
    required_cols = _required_columns()
    dfs: list[pd.DataFrame] = []
    for year in years:
        path = f"{_DATASET_DIR}/all_{year}.parquet"
        if not os.path.exists(path):
            continue
        available = set(pq.ParquetFile(path).schema.names)
        cols_to_read = [c for c in required_cols if c in available]
        dfs.append(pd.read_parquet(path, columns=cols_to_read))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Cleanup + safety caps
    if "original_price" in df.columns:
        df = df[df["original_price"] < 10e5]

    df = df.replace([np.inf, -np.inf], np.nan).infer_objects()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "c0" in df.columns:
        df = df.dropna(subset=["c0"])

    # Batch-compute derived columns to avoid DataFrame fragmentation
    new_cols: dict[str, pd.Series | object] = {}

    # event_price
    if "original_price" in df.columns and "c0" in df.columns:
        new_cols["event_price"] = df["original_price"].where(df["original_price"].notna(), df["c0"])
    elif "c0" in df.columns:
        new_cols["event_price"] = df["c0"]
    else:
        new_cols["event_price"] = np.nan

    # event_move
    if "cpct0" in df.columns and "atrp200" in df.columns:
        new_cols["event_move"] = df["cpct0"] / df["atrp200"]
    else:
        new_cols["event_move"] = 0.0

    new_cols["direction"] = np.sign(new_cols["event_move"]).replace(0, 1) if isinstance(new_cols["event_move"], pd.Series) else 1

    # Boolean event flags — original + new
    _event_flags = (
        "is_earnings", "is_etf",
        "evt_atrp_breakout", "evt_green_line_breakout", "evt_bb_lower_touch",
        "evt_episodic_pivot", "evt_pre_earnings", "evt_ema_reclaim", "evt_selloff",
    )
    for c in _event_flags:
        new_cols[c] = df[c].fillna(False).astype(bool) if c in df.columns else False

    # SPY Context
    direction = new_cols["direction"]
    if "spy0" in df.columns and "spy5" in df.columns:
        spy_change = df["spy5"] - df["spy0"]
        aligned_spy = spy_change * direction
        conditions = [aligned_spy > 0.5, aligned_spy < -0.5]
        new_cols["spy_class"] = np.select(conditions, ["Supporting", "Non-Supporting"], default="Neutral")
    else:
        new_cols["spy_class"] = "Unknown"

    # Drop original boolean columns that are being replaced, then concat all new columns at once
    bool_cols_to_replace = [c for c in _event_flags if c in df.columns]
    df = df.drop(columns=bool_cols_to_replace)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def load_earnings_events(
    years: range,
    *,
    min_sue: float | None = None,
    max_sue: float | None = None,
    direction: str | None = None,
) -> pd.DataFrame:
    """
    Load dataset filtered to earnings events only, with optional SUE filters.

    Parameters
    ----------
    years:
        Year range to load (e.g. range(2016, 2026)).
    min_sue:
        Minimum SUE (inclusive). Use for beat filtering (e.g. min_sue=0).
    max_sue:
        Maximum SUE (inclusive). Use for miss filtering (e.g. max_sue=0).
    direction:
        'beat', 'miss', or None for all.

    Returns DataFrame with only earnings events and all forward return columns.
    """
    df = load_and_prep_data(years)
    if df.empty:
        return df

    mask = df["is_earnings"].fillna(False).astype(bool)

    if "sue" in df.columns:
        if min_sue is not None:
            mask &= df["sue"] >= min_sue
        if max_sue is not None:
            mask &= df["sue"] <= max_sue

    if direction is not None and "surprise_dir" in df.columns:
        mask &= df["surprise_dir"] == direction

    return df.loc[mask].reset_index(drop=True)
