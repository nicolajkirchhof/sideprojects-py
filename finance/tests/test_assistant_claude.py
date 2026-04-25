"""
Tests for TA-E5-S2 — On-demand Claude candidate analysis.

Covers:
  - CandidateAnalysis model
  - analyze_candidate() with mocked Claude API
  - CandidateAnalysisThread signal behaviour
  - Cache I/O for per-symbol analyses
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from unittest.mock import patch

import pytest

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)

_MOCK_ROW = {
    "symbol": "AAPL",
    "direction": "long",
    "price": 185.0,
    "change_5d_pct": 5.2,
    "change_1m_pct": 12.3,
    "rvol_20d": 1.8,
    "atr_pct_20d": 3.1,
    "score_total": 75.0,
    "sector": "Technology",
    "iv_percentile": 42.0,
    "put_call_vol_5d": 0.6,
    "earnings_surprise_pct": 8.3,
    "latest_earnings": "2026-05-01",
    "tags": ["52w-high", "vol-spike"],
    "dimensions": [
        {"dimension": 1, "weighted_score": 20.0},
        {"dimension": 2, "weighted_score": 18.0},
    ],
}

_MOCK_ANALYSIS_JSON = json.dumps({
    "setup_type": "Type A — EP",
    "profit_mechanism": "PM-02 PEAD",
    "thesis": "Strong earnings beat with institutional follow-through expected.",
    "entry": 186.50,
    "stop": 173.50,
    "target": 212.00,
    "confidence": "HIGH",
})


# ---------------------------------------------------------------------------
# CandidateAnalysis model
# ---------------------------------------------------------------------------


def test_candidate_analysis_defaults():
    from finance.apps.assistant._models import CandidateAnalysis

    a = CandidateAnalysis()
    assert a.setup_type == ""
    assert a.confidence == ""
    assert a.entry is None
    assert a.raw_response == ""


def test_candidate_analysis_to_dict_round_trip():
    """dataclasses.asdict / CandidateAnalysis(**d) must round-trip all fields."""
    import dataclasses
    from finance.apps.assistant._models import CandidateAnalysis

    a = CandidateAnalysis(
        setup_type="Type A — EP",
        profit_mechanism="PM-02 PEAD",
        thesis="EP trade",
        entry=186.50,
        stop=173.50,
        target=212.00,
        confidence="HIGH",
        raw_response="raw",
    )
    d = dataclasses.asdict(a)
    a2 = CandidateAnalysis(**d)
    assert a2.setup_type == "Type A — EP"
    assert a2.entry == 186.50
    assert a2.raw_response == "raw"


# ---------------------------------------------------------------------------
# analyze_candidate()
# ---------------------------------------------------------------------------


def test_analyze_candidate_parses_json_response():
    """Happy path: Claude returns valid JSON → parsed into CandidateAnalysis."""
    from finance.apps.assistant._claude import analyze_candidate

    with patch("finance.apps.assistant._claude._call_claude", return_value=_MOCK_ANALYSIS_JSON):
        result = analyze_candidate(row=_MOCK_ROW, model="claude-sonnet-4-6")

    assert result.setup_type == "Type A — EP"
    assert result.profit_mechanism == "PM-02 PEAD"
    assert result.entry == 186.50
    assert result.stop == 173.50
    assert result.confidence == "HIGH"


def test_analyze_candidate_stores_raw_response():
    """raw_response is always set to the full Claude output."""
    from finance.apps.assistant._claude import analyze_candidate

    with patch("finance.apps.assistant._claude._call_claude", return_value=_MOCK_ANALYSIS_JSON):
        result = analyze_candidate(row=_MOCK_ROW, model="claude-sonnet-4-6")

    assert result.raw_response == _MOCK_ANALYSIS_JSON


def test_analyze_candidate_handles_unparseable_response():
    """If Claude returns garbage, raw_response is set but fields stay empty."""
    from finance.apps.assistant._claude import analyze_candidate

    garbage = "I cannot analyse this trade."
    with patch("finance.apps.assistant._claude._call_claude", return_value=garbage):
        result = analyze_candidate(row=_MOCK_ROW, model="claude-sonnet-4-6")

    assert result.setup_type == ""
    assert result.raw_response == garbage


def test_analyze_candidate_handles_api_failure():
    """If _call_claude returns empty string, return empty CandidateAnalysis."""
    from finance.apps.assistant._claude import analyze_candidate

    with patch("finance.apps.assistant._claude._call_claude", return_value=""):
        result = analyze_candidate(row=_MOCK_ROW, model="claude-sonnet-4-6")

    assert result.setup_type == ""
    assert result.raw_response == ""


# ---------------------------------------------------------------------------
# Cache I/O — per-symbol candidate_analyses
# ---------------------------------------------------------------------------


def test_update_cache_candidate_analysis_writes_entry(tmp_path):
    """update_cache_candidate_analysis writes the analysis under candidate_analyses.SYMBOL."""
    from finance.apps.assistant._pipeline import update_cache_candidate_analysis, write_cache

    write_cache([], date(2026, 4, 23), base_dir=tmp_path)
    analysis_dict = {"setup_type": "Type A", "confidence": "HIGH", "entry": 186.5,
                     "stop": 173.5, "target": 212.0, "profit_mechanism": "PM-02",
                     "thesis": "EP", "raw_response": ""}
    update_cache_candidate_analysis(date(2026, 4, 23), "AAPL", analysis_dict, base_dir=tmp_path)

    import json
    from finance.apps.assistant._pipeline import cache_path
    payload = json.loads(cache_path(date(2026, 4, 23), base_dir=tmp_path).read_text())
    assert payload["candidate_analyses"]["AAPL"]["confidence"] == "HIGH"


def test_read_candidate_analysis_from_cache_returns_dict(tmp_path):
    """read_candidate_analysis_from_cache returns the stored dict."""
    from finance.apps.assistant._pipeline import (
        read_candidate_analysis_from_cache,
        update_cache_candidate_analysis,
        write_cache,
    )

    write_cache([], date(2026, 4, 23), base_dir=tmp_path)
    analysis_dict = {"setup_type": "Type B", "confidence": "MEDIUM", "entry": 100.0,
                     "stop": 93.0, "target": 120.0, "profit_mechanism": "PM-01",
                     "thesis": "VCP", "raw_response": ""}
    update_cache_candidate_analysis(date(2026, 4, 23), "TSLA", analysis_dict, base_dir=tmp_path)

    result = read_candidate_analysis_from_cache(date(2026, 4, 23), "TSLA", base_dir=tmp_path)
    assert result is not None
    assert result["setup_type"] == "Type B"


def test_read_candidate_analysis_from_cache_returns_none_when_absent(tmp_path):
    """Returns None when no cache file exists."""
    from finance.apps.assistant._pipeline import read_candidate_analysis_from_cache

    result = read_candidate_analysis_from_cache(date(2026, 4, 23), "AAPL", base_dir=tmp_path)
    assert result is None


def test_read_candidate_analysis_from_cache_returns_none_for_unknown_symbol(tmp_path):
    """Returns None when cache exists but symbol not yet analysed."""
    from finance.apps.assistant._pipeline import (
        read_candidate_analysis_from_cache,
        write_cache,
    )

    write_cache([], date(2026, 4, 23), base_dir=tmp_path)
    result = read_candidate_analysis_from_cache(date(2026, 4, 23), "UNKNOWN", base_dir=tmp_path)
    assert result is None


def test_update_cache_candidate_analysis_noop_when_no_cache(tmp_path):
    """update_cache_candidate_analysis is a no-op when no cache file exists."""
    from finance.apps.assistant._pipeline import update_cache_candidate_analysis

    # Should not raise
    update_cache_candidate_analysis(date(2026, 4, 23), "AAPL",
                                     {"setup_type": "Type A"}, base_dir=tmp_path)


# ---------------------------------------------------------------------------
# CandidateAnalysisThread — Qt integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_candidate_analysis_thread_emits_analysis_ready():
    """Thread emits analysis_ready with a dict when analyze_candidate succeeds."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._pipeline import CandidateAnalysisThread

    ensure_qt_app()

    received: list[dict] = []

    with patch("finance.apps.assistant._pipeline.analyze_candidate") as mock_ac:
        from finance.apps.assistant._models import CandidateAnalysis
        mock_ac.return_value = CandidateAnalysis(setup_type="Type A", confidence="HIGH")

        thread = CandidateAnalysisThread(row=_MOCK_ROW, model="claude-sonnet-4-6")
        thread.analysis_ready.connect(received.append)
        thread.start()
        thread.wait(5000)

    from pyqtgraph.Qt import QtWidgets
    QtWidgets.QApplication.processEvents()

    assert len(received) == 1
    assert received[0]["setup_type"] == "Type A"
    assert received[0]["confidence"] == "HIGH"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_candidate_analysis_thread_emits_error_on_failure():
    """Thread emits error signal when analyze_candidate raises."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._pipeline import CandidateAnalysisThread

    ensure_qt_app()

    errors: list[str] = []

    with patch("finance.apps.assistant._pipeline.analyze_candidate",
               side_effect=RuntimeError("Claude timeout")):
        thread = CandidateAnalysisThread(row=_MOCK_ROW, model="claude-sonnet-4-6")
        thread.error.connect(errors.append)
        thread.start()
        thread.wait(5000)

    from pyqtgraph.Qt import QtWidgets
    QtWidgets.QApplication.processEvents()

    assert len(errors) == 1
    assert "Claude timeout" in errors[0]


# ---------------------------------------------------------------------------
# TopNAnalysisThread — Qt integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_top_n_analysis_thread_emits_per_row():
    """Thread emits row_analysis_ready once per row in top_n."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._pipeline import TopNAnalysisThread

    ensure_qt_app()

    rows = [
        {**_MOCK_ROW, "symbol": "AAPL"},
        {**_MOCK_ROW, "symbol": "MSFT"},
        {**_MOCK_ROW, "symbol": "NVDA"},
    ]
    received: list[tuple[str, dict]] = []

    with patch("finance.apps.assistant._pipeline.analyze_candidate") as mock_ac:
        from finance.apps.assistant._models import CandidateAnalysis
        mock_ac.return_value = CandidateAnalysis(setup_type="Type A", confidence="HIGH")

        thread = TopNAnalysisThread(rows=rows, model="claude-haiku-4-5-20251001", top_n=2)
        thread.row_analysis_ready.connect(lambda sym, d: received.append((sym, d)))
        thread.start()
        thread.wait(5000)

    from pyqtgraph.Qt import QtWidgets
    QtWidgets.QApplication.processEvents()

    assert len(received) == 2
    assert received[0][0] == "AAPL"
    assert received[1][0] == "MSFT"
    assert received[0][1]["setup_type"] == "Type A"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_top_n_analysis_thread_continues_on_per_row_error():
    """A per-row API failure must not stop remaining rows from being analysed."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._pipeline import TopNAnalysisThread

    ensure_qt_app()

    rows = [
        {**_MOCK_ROW, "symbol": "AAPL"},
        {**_MOCK_ROW, "symbol": "MSFT"},
    ]
    received: list[str] = []

    def _flaky(row, model):
        from finance.apps.assistant._models import CandidateAnalysis
        if row.get("symbol") == "AAPL":
            raise RuntimeError("API error for AAPL")
        return CandidateAnalysis(setup_type="Type B")

    with patch("finance.apps.assistant._pipeline.analyze_candidate", side_effect=_flaky):
        thread = TopNAnalysisThread(rows=rows, model="claude-haiku-4-5-20251001", top_n=2)
        thread.row_analysis_ready.connect(lambda sym, d: received.append(sym))
        thread.start()
        thread.wait(5000)

    from pyqtgraph.Qt import QtWidgets
    QtWidgets.QApplication.processEvents()

    # AAPL failed silently; MSFT must still be emitted
    assert received == ["MSFT"]
