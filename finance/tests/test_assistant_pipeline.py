"""
Tests for finance.apps.assistant pipeline thread and cache I/O — TA-E2-S2.

Tests are written first (TDD).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from finance.apps.analyst._models import Candidate, EnrichedCandidate, TechnicalData
from finance.apps.assistant._models import (
    CandidateScore,
    ComponentScore,
    DimensionScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_DATE = date(2026, 4, 23)


def _make_candidate(symbol: str = "AAPL") -> Candidate:
    return Candidate(
        symbol=symbol,
        price=150.0,
        change_5d_pct=5.2,
        rvol_20d=1.8,
        iv_percentile=45.0,
        sector="Technology",
        latest_earnings="2026-06-01",
        market_cap_k=2_800_000_000_000,
    )


def _make_enriched(symbol: str = "AAPL") -> EnrichedCandidate:
    return EnrichedCandidate(
        candidate=_make_candidate(symbol),
        technicals=TechnicalData(sma_50=140.0, sma_200=120.0),
        data_available=True,
    )


def _make_score(direction: str = "long") -> CandidateScore:
    dim = DimensionScore(
        dimension=1,
        name="Trend Template",
        raw_score=0.8,
        weighted_score=20.0,
        components=[ComponentScore(name="Price vs 50d SMA", raw_score=1.0, available=True, source="scanner")],
        hard_gate_fired=False,
        partial=False,
    )
    return CandidateScore(
        direction=direction,
        dimensions=[dim],
        tag_bonus=4.0,
        total=24.0,
        tags=["52w-high", "vol-spike"],
    )


# ---------------------------------------------------------------------------
# Cache I/O tests
# ---------------------------------------------------------------------------


def test_cache_path_format(tmp_path):
    from finance.apps.assistant._pipeline import cache_path
    p = cache_path(_TEST_DATE, base_dir=tmp_path)
    assert p.name == "2026-04-23.json"
    assert p.parent == tmp_path


def test_write_cache_creates_file(tmp_path):
    from finance.apps.assistant._pipeline import write_cache, cache_path
    rows = [{"symbol": "AAPL", "score_total": 72.5}]
    written = write_cache(rows, _TEST_DATE, base_dir=tmp_path)
    assert written.exists()
    assert written == cache_path(_TEST_DATE, base_dir=tmp_path)


def test_write_cache_valid_json(tmp_path):
    from finance.apps.assistant._pipeline import write_cache
    rows = [{"symbol": "AAPL", "score_total": 72.5}]
    path = write_cache(rows, _TEST_DATE, base_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-04-23"
    assert payload["rows"] == rows
    assert "created_at" in payload


def test_read_cache_returns_none_when_missing(tmp_path):
    from finance.apps.assistant._pipeline import read_cache
    result = read_cache(_TEST_DATE, base_dir=tmp_path)
    assert result is None


def test_read_cache_round_trip(tmp_path):
    from finance.apps.assistant._pipeline import write_cache, read_cache
    rows = [{"symbol": "AAPL", "score_total": 72.5}, {"symbol": "MSFT", "score_total": 60.0}]
    write_cache(rows, _TEST_DATE, base_dir=tmp_path)
    loaded = read_cache(_TEST_DATE, base_dir=tmp_path)
    assert loaded == rows


# ---------------------------------------------------------------------------
# build_result_row tests
# ---------------------------------------------------------------------------


def test_build_result_row_contains_symbol():
    from finance.apps.assistant._pipeline import build_result_row
    row = build_result_row(_make_enriched("TSLA"), _make_score())
    assert row["symbol"] == "TSLA"


def test_build_result_row_contains_direction():
    from finance.apps.assistant._pipeline import build_result_row
    row = build_result_row(_make_enriched(), _make_score("short"))
    assert row["direction"] == "short"


def test_build_result_row_contains_score_total():
    from finance.apps.assistant._pipeline import build_result_row
    row = build_result_row(_make_enriched(), _make_score())
    assert row["score_total"] == pytest.approx(24.0)


def test_build_result_row_contains_tags():
    from finance.apps.assistant._pipeline import build_result_row
    row = build_result_row(_make_enriched(), _make_score())
    assert row["tags"] == ["52w-high", "vol-spike"]


def test_build_result_row_contains_dimensions():
    from finance.apps.assistant._pipeline import build_result_row
    row = build_result_row(_make_enriched(), _make_score())
    assert isinstance(row["dimensions"], list)
    assert len(row["dimensions"]) == 1
    assert row["dimensions"][0]["dimension"] == 1


def test_build_result_row_candidate_fields_present():
    from finance.apps.assistant._pipeline import build_result_row
    row = build_result_row(_make_enriched(), _make_score())
    assert row["price"] == pytest.approx(150.0)
    assert row["sector"] == "Technology"
    assert row["change_5d_pct"] == pytest.approx(5.2)
    assert row["rvol_20d"] == pytest.approx(1.8)


def test_build_result_row_is_json_serialisable():
    """Row must survive a JSON round-trip (no non-serialisable types)."""
    from finance.apps.assistant._pipeline import build_result_row
    row = build_result_row(_make_enriched(), _make_score())
    serialised = json.dumps(row)
    loaded = json.loads(serialised)
    assert loaded["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# check_ibkr_gateway tests
# ---------------------------------------------------------------------------


def test_check_ibkr_gateway_raises_on_closed_port():
    """Connecting to a port that is definitely closed should raise RuntimeError."""
    from finance.apps.assistant._pipeline import check_ibkr_gateway
    with pytest.raises(RuntimeError, match="IBKR Gateway"):
        # Port 1 is almost certainly not open on any dev machine
        check_ibkr_gateway(host="127.0.0.1", port=1, timeout=0.5)


# ---------------------------------------------------------------------------
# PipelineThread import + attribute tests (no Qt instantiation needed)
# ---------------------------------------------------------------------------


def test_pipeline_thread_module_importable():
    from finance.apps.assistant import _pipeline  # noqa: F401
    assert hasattr(_pipeline, "PipelineThread")


def test_pipeline_thread_has_signals():
    from finance.apps.assistant._pipeline import PipelineThread
    assert hasattr(PipelineThread, "stage_changed")
    assert hasattr(PipelineThread, "candidate_count_changed")
    assert hasattr(PipelineThread, "finished_ok")
    assert hasattr(PipelineThread, "error")


# ---------------------------------------------------------------------------
# ErrorDialog import test
# ---------------------------------------------------------------------------


def test_error_dialog_module_importable():
    from finance.apps.assistant import _error_dialog  # noqa: F401
    assert hasattr(_error_dialog, "ErrorDialog")
    assert hasattr(_error_dialog, "show_pipeline_error")


# ---------------------------------------------------------------------------
# Window integration tests (require display)
# ---------------------------------------------------------------------------

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_has_results_store():
    """_results starts empty regardless of any on-disk cache."""
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._window import AssistantWindow

    app = ensure_qt_app()
    apply_dark_palette(app)
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=None):
        win = AssistantWindow()
    assert hasattr(win, "_results")
    assert win._results == []
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_export_buttons_disabled_initially():
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._window import AssistantWindow

    app = ensure_qt_app()
    apply_dark_palette(app)
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=None):
        win = AssistantWindow()
    assert not win._btn_export_bc.isEnabled()
    assert not win._btn_export_tws.isEnabled()
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_run_button_click_starts_pipeline_thread():
    """Clicking Run Pipeline must not raise — verifies the lambda wrapper is correct."""
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._pipeline import PipelineThread
    from finance.apps.assistant._window import AssistantWindow

    app = ensure_qt_app()
    apply_dark_palette(app)
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=None):
        win = AssistantWindow()

    started: list[bool] = []

    def _fake_start(self_thread: PipelineThread) -> None:
        started.append(True)

    with patch.object(PipelineThread, "start", _fake_start):
        win._btn_run.click()  # would raise TypeError before the lambda fix

    assert started, "PipelineThread.start() was not called"
    win.close()
