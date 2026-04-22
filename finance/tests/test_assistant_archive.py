"""
Tests for finance.apps.assistant._archive — daily Parquet archive I/O.

Tests are written first (TDD). They define expected behaviour before
the implementation modules exist.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from finance.apps.assistant._archive import (
    COLUMN_ORDER,
    DTYPES,
    archive_path,
    candidates_to_df,
    parse_tags,
    read_archive,
    read_candidates,
    read_market,
    write_archive,
)
from finance.apps.analyst._models import Candidate, EnrichedCandidate, TechnicalData
from finance.apps.assistant._models import (
    CandidateScore,
    ComponentScore,
    DimensionScore,
    ScoringConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_DATE = date(2026, 4, 22)

_FLOAT_COLS = {c for c, t in DTYPES.items() if t == "Float64"}
_STRING_COLS = {c for c, t in DTYPES.items() if t == "string"}


def _make_dimension(n: int, score: float = 0.8) -> DimensionScore:
    weight = {1: 25, 2: 25, 3: 15, 4: 20, 5: 15}[n]
    return DimensionScore(
        dimension=n,
        name=f"D{n}",
        raw_score=score,
        weighted_score=score * weight,
        components=[],
    )


def _make_score(total: float = 72.0, direction: str = "long") -> CandidateScore:
    dims = [_make_dimension(n) for n in range(1, 6)]
    return CandidateScore(
        direction=direction,
        dimensions=dims,
        tag_bonus=4.0,
        total=total,
        tags=["vol-spike", "trend-seeker"],
    )


def _make_candidate(**kwargs) -> Candidate:
    defaults = {
        "symbol": "TEST",
        "price": 100.0,
        "change_pct": 1.5,
        "change_5d_pct": 4.0,
        "change_1m_pct": 8.0,
        "rvol_20d": 1.3,
        "atr_pct_20d": 3.5,
        "sector": "Technology",
        "source_scanner": "long-universe",
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


def _make_enriched(candidate: Candidate | None = None) -> EnrichedCandidate:
    c = candidate or _make_candidate()
    return EnrichedCandidate(candidate=c, technicals=None, data_available=False)


def _make_archive_df(n_candidates: int = 2, n_market: int = 2) -> pd.DataFrame:
    """Build a minimal well-formed archive DataFrame for I/O tests."""
    rows = []
    for i in range(n_candidates):
        row = {col: pd.NA for col in COLUMN_ORDER}
        row["date"] = _TEST_DATE.isoformat()
        row["row_type"] = "candidate"
        row["symbol"] = f"TICK{i}"
        row["category"] = pd.NA
        row["price"] = float(100 + i)
        row["score_total"] = float(70 + i)
        row["score_direction"] = "long"
        row["score_tags"] = "vol-spike,trend-seeker"
        rows.append(row)
    for j in range(n_market):
        row = {col: pd.NA for col in COLUMN_ORDER}
        row["date"] = _TEST_DATE.isoformat()
        row["row_type"] = "market"
        row["symbol"] = ["SPY", "QQQ", "GLD", "TLT"][j % 4]
        row["category"] = "Indices"
        row["price"] = float(500 + j * 10)
        row["open"] = float(498 + j * 10)
        row["hv20"] = 14.5
        rows.append(row)
    df = pd.DataFrame(rows)[COLUMN_ORDER]
    return df.astype(DTYPES)


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_column_count(self) -> None:
        assert len(COLUMN_ORDER) == 62

    def test_required_identity_columns(self) -> None:
        for col in ("date", "row_type", "symbol", "category"):
            assert col in COLUMN_ORDER

    def test_required_score_columns(self) -> None:
        for col in (
            "score_total", "score_d1_weighted", "score_d5_weighted",
            "score_direction", "score_tags", "score_tag_bonus",
        ):
            assert col in COLUMN_ORDER

    def test_required_market_extra_columns(self) -> None:
        for col in ("hv20", "iv_rank", "open", "high", "low", "gap_pct"):
            assert col in COLUMN_ORDER

    def test_text_columns_present(self) -> None:
        assert "ai_analysis" in COLUMN_ORDER
        assert "trigger_event" in COLUMN_ORDER

    def test_dtypes_cover_all_columns(self) -> None:
        assert set(DTYPES.keys()) == set(COLUMN_ORDER)

    def test_float_columns_use_nullable_float(self) -> None:
        # score_total, price etc. must use Float64 (nullable), not float64
        for col in ("score_total", "price", "hv20", "change_1m_pct"):
            assert DTYPES[col] == "Float64", f"{col} should be Float64"

    def test_string_columns_use_nullable_string(self) -> None:
        for col in ("symbol", "row_type", "score_direction", "ai_analysis", "ttm_squeeze"):
            assert DTYPES[col] == "string", f"{col} should be string"

    def test_no_duplicate_columns(self) -> None:
        assert len(COLUMN_ORDER) == len(set(COLUMN_ORDER))


# ---------------------------------------------------------------------------
# 2. archive_path
# ---------------------------------------------------------------------------

class TestArchivePath:
    def test_filename_format(self) -> None:
        p = archive_path(_TEST_DATE)
        assert p.name == "2026-04-22.parquet"

    def test_path_is_under_data_dir(self) -> None:
        p = archive_path(_TEST_DATE)
        assert "assistant" in str(p)


# ---------------------------------------------------------------------------
# 3. write / read round-trip
# ---------------------------------------------------------------------------

class TestWriteRead:
    def test_round_trip_preserves_values(self, tmp_path: Path) -> None:
        df = _make_archive_df(n_candidates=3, n_market=2)
        write_archive(df, _TEST_DATE, data_dir=tmp_path)
        result = read_archive(_TEST_DATE, data_dir=tmp_path)
        assert len(result) == 5
        # Candidate price preserved
        cand = result[result["row_type"] == "candidate"]
        assert set(cand["symbol"].tolist()) == {"TICK0", "TICK1", "TICK2"}

    def test_round_trip_preserves_na(self, tmp_path: Path) -> None:
        df = _make_archive_df()
        write_archive(df, _TEST_DATE, data_dir=tmp_path)
        result = read_archive(_TEST_DATE, data_dir=tmp_path)
        # candidate rows should have NA for market-only columns
        cand = result[result["row_type"] == "candidate"]
        assert cand["hv20"].isna().all()
        assert cand["open"].isna().all()

    def test_round_trip_dtypes(self, tmp_path: Path) -> None:
        df = _make_archive_df()
        write_archive(df, _TEST_DATE, data_dir=tmp_path)
        result = read_archive(_TEST_DATE, data_dir=tmp_path)
        # Float64 dtype preserved
        assert str(result["price"].dtype) == "Float64"
        # string dtype preserved
        assert str(result["symbol"].dtype) == "string"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        df = _make_archive_df()
        write_archive(df, _TEST_DATE, data_dir=nested)
        assert archive_path(_TEST_DATE, data_dir=nested).exists()

    def test_overwrite_replaces_file(self, tmp_path: Path) -> None:
        df1 = _make_archive_df(n_candidates=2, n_market=0)
        write_archive(df1, _TEST_DATE, data_dir=tmp_path)
        df2 = _make_archive_df(n_candidates=5, n_market=0)
        write_archive(df2, _TEST_DATE, data_dir=tmp_path)
        result = read_archive(_TEST_DATE, data_dir=tmp_path)
        assert len(result) == 5  # not 7 (old + new)

    def test_write_returns_path(self, tmp_path: Path) -> None:
        df = _make_archive_df()
        path = write_archive(df, _TEST_DATE, data_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".parquet"


# ---------------------------------------------------------------------------
# 4. Missing file handling
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_read_absent_date_returns_empty_df(self, tmp_path: Path) -> None:
        result = read_archive(date(2000, 1, 1), data_dir=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_df_has_correct_columns(self, tmp_path: Path) -> None:
        result = read_archive(date(2000, 1, 1), data_dir=tmp_path)
        assert set(result.columns) == set(COLUMN_ORDER)


# ---------------------------------------------------------------------------
# 5. Row type filters
# ---------------------------------------------------------------------------

class TestRowTypeFilters:
    def test_read_candidates_filters_correctly(self, tmp_path: Path) -> None:
        df = _make_archive_df(n_candidates=3, n_market=2)
        write_archive(df, _TEST_DATE, data_dir=tmp_path)
        result = read_candidates(_TEST_DATE, data_dir=tmp_path)
        assert len(result) == 3
        assert (result["row_type"] == "candidate").all()

    def test_read_market_filters_correctly(self, tmp_path: Path) -> None:
        df = _make_archive_df(n_candidates=3, n_market=2)
        write_archive(df, _TEST_DATE, data_dir=tmp_path)
        result = read_market(_TEST_DATE, data_dir=tmp_path)
        assert len(result) == 2
        assert (result["row_type"] == "market").all()

    def test_read_candidates_empty_on_missing_file(self, tmp_path: Path) -> None:
        result = read_candidates(date(2000, 1, 1), data_dir=tmp_path)
        assert len(result) == 0

    def test_read_market_empty_on_missing_file(self, tmp_path: Path) -> None:
        result = read_market(date(2000, 1, 1), data_dir=tmp_path)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 6. candidates_to_df
# ---------------------------------------------------------------------------

class TestCandidatesToDf:
    def test_one_candidate_produces_one_row(self) -> None:
        ec = _make_enriched()
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert len(df) == 1

    def test_row_type_is_candidate(self) -> None:
        ec = _make_enriched()
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert df.iloc[0]["row_type"] == "candidate"

    def test_symbol_matches_candidate(self) -> None:
        ec = _make_enriched(_make_candidate(symbol="NVDA"))
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert df.iloc[0]["symbol"] == "NVDA"

    def test_score_total_populated(self) -> None:
        ec = _make_enriched()
        score = _make_score(total=85.5)
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert float(df.iloc[0]["score_total"]) == pytest.approx(85.5)

    def test_score_tags_is_comma_joined_string(self) -> None:
        ec = _make_enriched()
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert df.iloc[0]["score_tags"] == "vol-spike,trend-seeker"

    def test_score_d1_through_d5_populated(self) -> None:
        ec = _make_enriched()
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        for n in range(1, 6):
            col = f"score_d{n}_weighted"
            assert pd.notna(df.iloc[0][col]), f"{col} should not be NA"

    def test_market_only_columns_are_na(self) -> None:
        ec = _make_enriched()
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        for col in ("hv20", "iv_rank", "open", "high", "low", "gap_pct"):
            assert pd.isna(df.iloc[0][col]), f"{col} should be NA for candidate"

    def test_category_is_na_for_candidates(self) -> None:
        ec = _make_enriched()
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert pd.isna(df.iloc[0]["category"])

    def test_all_columns_present(self) -> None:
        ec = _make_enriched()
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert set(df.columns) == set(COLUMN_ORDER)

    def test_date_column_populated(self) -> None:
        ec = _make_enriched()
        score = _make_score()
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert df.iloc[0]["date"] == "2026-04-22"

    def test_empty_tags_gives_empty_string(self) -> None:
        ec = _make_enriched()
        score = CandidateScore(
            direction="long",
            dimensions=[_make_dimension(n) for n in range(1, 6)],
            tag_bonus=0.0,
            total=60.0,
            tags=[],
        )
        df = candidates_to_df([ec], [score], trade_date=_TEST_DATE)
        assert df.iloc[0]["score_tags"] == ""

    def test_multiple_candidates(self) -> None:
        ecs = [_make_enriched(_make_candidate(symbol=s)) for s in ["AAPL", "MSFT", "NVDA"]]
        scores = [_make_score() for _ in ecs]
        df = candidates_to_df(ecs, scores, trade_date=_TEST_DATE)
        assert len(df) == 3
        assert set(df["symbol"].tolist()) == {"AAPL", "MSFT", "NVDA"}


# ---------------------------------------------------------------------------
# 7. parse_tags
# ---------------------------------------------------------------------------

class TestParseTags:
    def test_parses_comma_joined_string(self) -> None:
        assert parse_tags("vol-spike,trend-seeker") == ["vol-spike", "trend-seeker"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert parse_tags("") == []

    def test_single_tag(self) -> None:
        assert parse_tags("pead-long") == ["pead-long"]

    def test_na_string_treated_as_no_tags(self) -> None:
        # pd.NA serialised as None or empty
        assert parse_tags(None) == []
