"""
finance.utils.earnings_data
============================
Earnings data loader — queries Dolt earnings DB directly for eps_history
and earnings_calendar. Computes SUE, surprise direction, and consecutive
beat streaks.

No caching layer — Dolt is assumed always available.

Usage
-----
    from finance.utils.earnings_data import load_earnings

    df = load_earnings("AAPL")
    # columns: symbol, date, when, period_end_date, eps, eps_est,
    #          sue, surprise_dir, consecutive_beats
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from finance.utils.dolt_data import db_earnings_connection, time_db_call


# ---------------------------------------------------------------------------
# Pure computation — no DB dependency, testable with synthetic data
# ---------------------------------------------------------------------------

def compute_earnings_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SUE, surprise_dir, and consecutive_beats to an earnings DataFrame.

    Expects columns: eps, eps_est (at minimum).
    Returns a copy with three new columns added.
    """
    if df.empty:
        out = df.copy()
        for col in ("sue", "surprise_dir", "consecutive_beats"):
            if col not in out.columns:
                out[col] = pd.Series(dtype=float if col != "surprise_dir" else object)
        return out

    df = df.sort_values("date").copy() if "date" in df.columns else df.copy()

    # SUE: (reported - estimate) / |estimate|
    eps = df["eps"] if "eps" in df.columns else pd.Series(np.nan, index=df.index)
    est = df["eps_est"] if "eps_est" in df.columns else pd.Series(np.nan, index=df.index)
    est_abs = est.abs()

    both_valid = eps.notna() & est.notna()
    nonzero_est = est_abs > 0

    df["sue"] = np.where(
        both_valid & nonzero_est,
        (eps - est) / est_abs,
        np.nan,
    )

    # surprise_dir: beat / miss / inline / unknown
    sue = df["sue"]
    df["surprise_dir"] = np.where(
        sue.isna(), "unknown",
        np.where(sue > 0, "beat",
                 np.where(sue < 0, "miss", "inline")),
    )

    # consecutive_beats: running count of consecutive beats, resets on non-beat
    running = 0
    beats = []
    for _, row in df.iterrows():
        if row["surprise_dir"] == "beat":
            running += 1
        else:
            running = 0
        beats.append(running)
    df["consecutive_beats"] = beats

    return df


# ---------------------------------------------------------------------------
# Dolt query
# ---------------------------------------------------------------------------

_CALENDAR_QUERY = text("""
    SELECT act_symbol AS symbol, date, `when`
    FROM earnings_calendar
    WHERE act_symbol = :symbol
    ORDER BY date
""")

_EPS_HISTORY_QUERY = text("""
    SELECT act_symbol AS symbol, period_end_date, reported AS eps, estimate AS eps_est
    FROM eps_history
    WHERE act_symbol = :symbol
    ORDER BY period_end_date
""")


def load_earnings(symbol: str) -> pd.DataFrame:
    """
    Load earnings history for a symbol from Dolt.

    Queries earnings_calendar and eps_history separately, then joins in
    pandas using merge_asof (most recent period before each announcement).

    Returns DataFrame with columns:
        symbol, date, when, period_end_date, eps, eps_est,
        sue, surprise_dir, consecutive_beats

    Returns empty DataFrame if no earnings found.
    """
    df_cal = pd.read_sql(_CALENDAR_QUERY, db_earnings_connection, params={"symbol": symbol})
    if df_cal.empty:
        return compute_earnings_fields(df_cal)

    df_eps = pd.read_sql(_EPS_HISTORY_QUERY, db_earnings_connection, params={"symbol": symbol})

    df_cal["date"] = pd.to_datetime(df_cal["date"])
    df_cal["when"] = (
        df_cal["when"]
        .str.replace("After market close", "post", regex=False)
        .str.replace("Before market open", "pre", regex=False)
    )

    if df_eps.empty:
        df_cal["period_end_date"] = pd.NaT
        df_cal["eps"] = np.nan
        df_cal["eps_est"] = np.nan
        return compute_earnings_fields(df_cal)

    df_eps["period_end_date"] = pd.to_datetime(df_eps["period_end_date"])
    for col in ("eps", "eps_est"):
        df_eps[col] = pd.to_numeric(df_eps[col], errors="coerce")

    # merge_asof: for each announcement date, find the most recent period_end_date before it
    df = pd.merge_asof(
        df_cal.sort_values("date"),
        df_eps[["period_end_date", "eps", "eps_est"]].sort_values("period_end_date"),
        left_on="date",
        right_on="period_end_date",
        direction="backward",
    )

    return compute_earnings_fields(df)
