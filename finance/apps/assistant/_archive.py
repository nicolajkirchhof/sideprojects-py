"""
finance.apps.assistant._archive
================================
Daily Parquet archive — unified schema for scored candidates and market rows.

One file per session date: _data/assistant/YYYY-MM-DD.parquet
All rows share the same 62-column schema; cross-type columns are pd.NA.

row_type='candidate'  — scored scanner candidates (scanner fields + scores)
row_type='market'     — market context instruments (OHLC + technicals, no scores)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from finance.apps.analyst._models import EnrichedCandidate
    from finance.apps.assistant._models import CandidateScore

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: Canonical column sequence. Every archive file has exactly these 62 columns.
COLUMN_ORDER: list[str] = [
    # --- Identity (4) ---
    "date",
    "row_type",
    "symbol",
    "category",
    # --- Shared: price + momentum (16) ---
    "price",
    "change_pct",
    "change_5d_pct",
    "change_1m_pct",
    "change_3m_pct",
    "change_6m_pct",
    "change_52w_pct",
    "volume",
    "rvol_20d",
    "atr_pct_20d",
    "pct_from_50d_sma",
    "slope_50d_sma",
    "slope_200d_sma",
    "bb_pct",
    "ttm_squeeze",
    "iv_percentile",
    # --- Market-only OHLC (4) ---
    "open",
    "high",
    "low",
    "gap_pct",
    # --- Market-only volatility extras (2) ---
    "hv20",
    "iv_rank",
    # --- Candidate-only scanner fields (25) ---
    "adr_pct_20d",
    "high_52w_distance_pct",
    "weighted_alpha",
    "perf_vs_market_5d",
    "perf_vs_market_1m",
    "perf_vs_market_3m",
    "gap_up_pct",
    "short_float",
    "short_interest_k",
    "short_interest_chg_pct",
    "days_to_cover",
    "earnings_surprise_pct",
    "earnings_surprise_q1",
    "earnings_surprise_q2",
    "earnings_surprise_q3",
    "put_call_vol_5d",
    "put_call_vol_1m",
    "iv_chg_5d",
    "iv_chg_1m",
    "vol_oi_ratio",
    "market_cap_k",
    "latest_earnings",
    "sector",
    "source_scanner",
    "trend_seeker_signal",
    # --- Score columns — candidates only (9) ---
    "score_direction",
    "score_total",
    "score_tag_bonus",
    "score_d1_weighted",
    "score_d2_weighted",
    "score_d3_weighted",
    "score_d4_weighted",
    "score_d5_weighted",
    "score_tags",
    # --- Future Claude text columns (2) ---
    "ai_analysis",
    "trigger_event",
]

assert len(COLUMN_ORDER) == 62, f"Expected 62 columns, got {len(COLUMN_ORDER)}"
assert len(COLUMN_ORDER) == len(set(COLUMN_ORDER)), "Duplicate column detected"

#: Pandas nullable dtypes for all columns. Float64 is nullable (supports pd.NA).
#: Use pd.ArrowDtype or pandas extension types — NOT numpy float64 — so Parquet
#: round-trips cleanly without converting pd.NA to NaN (which forces float).
DTYPES: dict[str, str] = {
    # identity
    "date":                    "string",
    "row_type":                "string",
    "symbol":                  "string",
    "category":                "string",
    # shared numeric
    "price":                   "Float64",
    "change_pct":              "Float64",
    "change_5d_pct":           "Float64",
    "change_1m_pct":           "Float64",
    "change_3m_pct":           "Float64",
    "change_6m_pct":           "Float64",
    "change_52w_pct":          "Float64",
    "volume":                  "Float64",
    "rvol_20d":                "Float64",
    "atr_pct_20d":             "Float64",
    "pct_from_50d_sma":        "Float64",
    "slope_50d_sma":           "Float64",
    "slope_200d_sma":          "Float64",
    "bb_pct":                  "Float64",
    "ttm_squeeze":             "string",
    "iv_percentile":           "Float64",
    # market OHLC
    "open":                    "Float64",
    "high":                    "Float64",
    "low":                     "Float64",
    "gap_pct":                 "Float64",
    # market volatility extras
    "hv20":                    "Float64",
    "iv_rank":                 "Float64",
    # candidate scanner
    "adr_pct_20d":             "Float64",
    "high_52w_distance_pct":   "Float64",
    "weighted_alpha":          "Float64",
    "perf_vs_market_5d":       "Float64",
    "perf_vs_market_1m":       "Float64",
    "perf_vs_market_3m":       "Float64",
    "gap_up_pct":              "Float64",
    "short_float":             "Float64",
    "short_interest_k":        "Float64",
    "short_interest_chg_pct":  "Float64",
    "days_to_cover":           "Float64",
    "earnings_surprise_pct":   "Float64",
    "earnings_surprise_q1":    "Float64",
    "earnings_surprise_q2":    "Float64",
    "earnings_surprise_q3":    "Float64",
    "put_call_vol_5d":         "Float64",
    "put_call_vol_1m":         "Float64",
    "iv_chg_5d":               "Float64",
    "iv_chg_1m":               "Float64",
    "vol_oi_ratio":            "Float64",
    "market_cap_k":            "Float64",
    "latest_earnings":         "string",
    "sector":                  "string",
    "source_scanner":          "string",
    "trend_seeker_signal":     "string",
    # scores
    "score_direction":         "string",
    "score_total":             "Float64",
    "score_tag_bonus":         "Float64",
    "score_d1_weighted":       "Float64",
    "score_d2_weighted":       "Float64",
    "score_d3_weighted":       "Float64",
    "score_d4_weighted":       "Float64",
    "score_d5_weighted":       "Float64",
    "score_tags":              "string",
    # text
    "ai_analysis":             "string",
    "trigger_event":           "string",
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_DEFAULT_ARCHIVE_DIR = Path("finance/_data/assistant")


def archive_path(trade_date: date, *, data_dir: Path | None = None) -> Path:
    """Return the Parquet path for a given session date."""
    base = Path(data_dir) if data_dir is not None else _DEFAULT_ARCHIVE_DIR
    return base / f"{trade_date.isoformat()}.parquet"


# ---------------------------------------------------------------------------
# Write / read
# ---------------------------------------------------------------------------

def write_archive(
    df: pd.DataFrame,
    trade_date: date,
    *,
    data_dir: Path | None = None,
) -> Path:
    """
    Write df to the daily archive Parquet file.

    The DataFrame is reindexed to COLUMN_ORDER and cast to DTYPES before
    writing. Overwrites any existing file for the same date.
    Returns the path written.
    """
    path = archive_path(trade_date, data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure correct column order and schema
    out = df.reindex(columns=COLUMN_ORDER)
    out = out.astype(DTYPES)

    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    import os
    os.replace(tmp, path)
    return path


def read_archive(
    trade_date: date,
    *,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Read the daily archive Parquet for a given date.

    Returns an empty DataFrame with the correct schema if the file is absent.
    """
    path = archive_path(trade_date, data_dir=data_dir)
    if not path.exists():
        return _empty_archive()
    return pd.read_parquet(path)


def read_candidates(
    trade_date: date,
    *,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Convenience: read archive filtered to row_type == 'candidate'."""
    df = read_archive(trade_date, data_dir=data_dir)
    if df.empty:
        return df
    return df[df["row_type"] == "candidate"].reset_index(drop=True)


def read_market(
    trade_date: date,
    *,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Convenience: read archive filtered to row_type == 'market'."""
    df = read_archive(trade_date, data_dir=data_dir)
    if df.empty:
        return df
    return df[df["row_type"] == "market"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Candidate serialisation
# ---------------------------------------------------------------------------

def candidates_to_df(
    candidates: list[EnrichedCandidate],
    scores: list[CandidateScore],
    *,
    trade_date: date,
) -> pd.DataFrame:
    """
    Flatten EnrichedCandidate + CandidateScore pairs into archive rows.

    Parameters
    ----------
    candidates:
        Enriched candidates in the same order as scores.
    scores:
        One CandidateScore per candidate.
    trade_date:
        Session date — written into the `date` column.

    Returns
    -------
    pd.DataFrame
        One row per candidate, all 62 columns, correct dtypes.
        Market-only columns (hv20, open, etc.) are pd.NA.
        Score columns are populated from CandidateScore.
    """
    rows = []
    date_str = trade_date.isoformat()
    for ec, sc in zip(candidates, scores):
        c = ec.candidate
        row: dict = {col: pd.NA for col in COLUMN_ORDER}

        # Identity
        row["date"] = date_str
        row["row_type"] = "candidate"
        row["symbol"] = c.symbol
        # category stays NA for candidates

        # Candidate scalar fields — direct mapping from Candidate dataclass
        _candidate_fields = [
            "price", "change_pct", "change_5d_pct", "change_1m_pct",
            "change_3m_pct", "change_6m_pct", "change_52w_pct", "volume",
            "rvol_20d", "atr_pct_20d", "pct_from_50d_sma", "slope_50d_sma",
            "slope_200d_sma", "bb_pct", "ttm_squeeze", "iv_percentile",
            "adr_pct_20d", "high_52w_distance_pct", "weighted_alpha",
            "perf_vs_market_5d", "perf_vs_market_1m", "perf_vs_market_3m",
            "gap_up_pct", "short_float", "short_interest_k",
            "short_interest_chg_pct", "days_to_cover", "earnings_surprise_pct",
            "earnings_surprise_q1", "earnings_surprise_q2", "earnings_surprise_q3",
            "put_call_vol_5d", "put_call_vol_1m", "iv_chg_5d", "iv_chg_1m",
            "vol_oi_ratio", "market_cap_k", "latest_earnings", "sector",
            "source_scanner", "trend_seeker_signal",
        ]
        for field in _candidate_fields:
            val = getattr(c, field, None)
            row[field] = val if val is not None else pd.NA

        # Score columns
        dims = {d.dimension: d for d in sc.dimensions}
        row["score_direction"] = sc.direction
        row["score_total"] = sc.total
        row["score_tag_bonus"] = sc.tag_bonus
        row["score_d1_weighted"] = dims[1].weighted_score if 1 in dims else pd.NA
        row["score_d2_weighted"] = dims[2].weighted_score if 2 in dims else pd.NA
        row["score_d3_weighted"] = dims[3].weighted_score if 3 in dims else pd.NA
        row["score_d4_weighted"] = dims[4].weighted_score if 4 in dims else pd.NA
        row["score_d5_weighted"] = dims[5].weighted_score if 5 in dims else pd.NA
        row["score_tags"] = ",".join(sc.tags)  # empty string when no tags

        rows.append(row)

    if not rows:
        return _empty_archive()

    df = pd.DataFrame(rows)[COLUMN_ORDER]
    return df.astype(DTYPES)


# ---------------------------------------------------------------------------
# parse_tags helper
# ---------------------------------------------------------------------------

def parse_tags(s: str | None) -> list[str]:
    """
    Deserialise the score_tags comma-joined string back to a list.

    Inverse of ``",".join(tags)`` used in candidates_to_df.
    Returns an empty list for None or empty string.
    """
    if not s:
        return []
    return s.split(",")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_archive() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema."""
    df = pd.DataFrame(columns=COLUMN_ORDER)
    return df.astype(DTYPES)
