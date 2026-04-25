"""
Tests for finance.apps.assistant._detail_panel — TA-E5-S1.

Written test-first (TDD).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)

# Minimal row — no sub-component fields (backwards-compat check)
_FULL_ROW = {
    "symbol": "AAPL",
    "direction": "long",
    "price": 185.0,
    "change_pct": 1.2,
    "change_5d_pct": 5.2,
    "change_1m_pct": 12.3,
    "rvol_20d": 1.8,
    "atr_pct_20d": 3.1,
    "volume": 85_000_000,
    "sector": "Technology",
    "market_cap_k": 2_900_000_000,
    "latest_earnings": "2026-05-01",
    "iv_percentile": 42.0,
    "put_call_vol_5d": 0.6,
    "earnings_surprise_pct": 8.3,
    "short_float": 1.2,
    "score_total": 75.0,
    "score_tag_bonus": 4.0,
    "tags": ["52w-high", "vol-spike"],
    "dimensions": [
        {"dimension": 1, "weighted_score": 20.0},
        {"dimension": 2, "weighted_score": 18.0},
        {"dimension": 3, "weighted_score": 12.0},
        {"dimension": 4, "weighted_score": 15.0},
        {"dimension": 5, "weighted_score": 10.0},
    ],
}

# Rich row — includes hard_gate, partial, and component sub-scores
_RICH_ROW = {
    **_FULL_ROW,
    "dimensions": [
        {
            "dimension": 1, "name": "Trend Template", "raw_score": 0.8,
            "weighted_score": 20.0, "hard_gate_fired": False, "partial": False,
            "components": [
                {"name": "price_vs_50sma", "raw_score": 1.0, "available": True, "source": "ibkr"},
                {"name": "sma50_slope", "raw_score": 0.5, "available": True, "source": "ibkr"},
            ],
        },
        {
            "dimension": 2, "name": "Relative Strength", "raw_score": 0.72,
            "weighted_score": 18.0, "hard_gate_fired": False, "partial": True,
            "components": [
                {"name": "rs_vs_spy", "raw_score": 0.9, "available": True, "source": "scanner"},
                {"name": "rs_peers", "raw_score": 0.0, "available": False, "source": "none"},
            ],
        },
        {
            "dimension": 3, "name": "Base Quality", "raw_score": 0.8,
            "weighted_score": 12.0, "hard_gate_fired": False, "partial": False,
            "components": [
                {"name": "consolidation", "raw_score": 0.8, "available": True, "source": "scanner"},
            ],
        },
        {
            "dimension": 4, "name": "Catalyst", "raw_score": 0.75,
            "weighted_score": 15.0, "hard_gate_fired": False, "partial": False,
            "components": [
                {"name": "eps_surprise", "raw_score": 0.8, "available": True, "source": "ibkr"},
            ],
        },
        {
            "dimension": 5, "name": "Risk", "raw_score": 0.0,
            "weighted_score": 0.0, "hard_gate_fired": True, "partial": False,
            "components": [
                {"name": "stop_distance", "raw_score": 0.0, "available": True, "source": "ibkr"},
            ],
        },
    ],
}


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_detail_panel_loads_without_error():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_detail_panel_shows_symbol():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    assert "AAPL" in panel._lbl_symbol.text()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_detail_panel_shows_score():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    assert "75" in panel._lbl_score.text()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_detail_panel_clears_on_none():
    """load_row(None) should clear the panel without error."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    panel.load_row(None)


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_right_panel_is_detail_panel():
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel
    from finance.apps.assistant._window import AssistantWindow

    ensure_qt_app()
    apply_dark_palette(ensure_qt_app())
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=None), \
         patch("finance.apps.assistant._window.load_daily", return_value=None):
        win = AssistantWindow()
    assert isinstance(win._right_panel, CandidateDetailPanel)
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_row_selection_updates_detail_panel():
    """Selecting a row in the watchlist loads that row's data into the detail panel."""
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._window import AssistantWindow

    ensure_qt_app()
    apply_dark_palette(ensure_qt_app())
    rows = [_FULL_ROW]
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=rows), \
         patch("finance.apps.assistant._window.load_daily", return_value=None):
        win = AssistantWindow()

    win._watchlist_table.selectRow(0)
    assert "AAPL" in win._right_panel._lbl_symbol.text()
    win.close()


# ---------------------------------------------------------------------------
# Hard gate + partial indicators (TA-E5-S1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_hard_gate_indicator_visible_when_fired():
    """Dimension with hard_gate_fired=True must show the ⚠ badge."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_RICH_ROW)

    # D5 has hard_gate_fired=True
    dim_row = panel._dim_rows[5]
    assert not dim_row.gate_label.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_hard_gate_indicator_hidden_when_not_fired():
    """Dimension with hard_gate_fired=False must NOT show the ⚠ badge."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_RICH_ROW)

    # D1 has hard_gate_fired=False
    dim_row = panel._dim_rows[1]
    assert dim_row.gate_label.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_partial_indicator_visible_when_partial():
    """Dimension with partial=True must show the ~ badge."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_RICH_ROW)

    # D2 has partial=True
    dim_row = panel._dim_rows[2]
    assert not dim_row.partial_label.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_partial_indicator_hidden_when_complete():
    """Dimension with partial=False must NOT show the ~ badge."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_RICH_ROW)

    # D1 has partial=False
    dim_row = panel._dim_rows[1]
    assert dim_row.partial_label.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_indicators_default_hidden_without_fields():
    """Old rows without hard_gate_fired/partial fields must not show indicators."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)

    for dim_row in panel._dim_rows.values():
        assert dim_row.gate_label.isHidden()
        assert dim_row.partial_label.isHidden()


# ---------------------------------------------------------------------------
# Sub-component expansion (TA-E5-S1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_components_hidden_by_default():
    """Sub-component detail is hidden before the expand button is clicked."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_RICH_ROW)

    dim_row = panel._dim_rows[1]
    assert dim_row.components_widget.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_expand_button_toggles_components():
    """Clicking the expand button shows sub-components; clicking again hides them."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_RICH_ROW)

    dim_row = panel._dim_rows[1]
    dim_row.expand_btn.click()
    assert not dim_row.components_widget.isHidden()

    dim_row.expand_btn.click()
    assert dim_row.components_widget.isHidden()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_components_hidden_when_no_components():
    """Expand button hidden when dimension has no component sub-scores."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)  # no components field

    # All expand buttons should be hidden (no data to show)
    for dim_row in panel._dim_rows.values():
        assert dim_row.expand_btn.isHidden()


# ---------------------------------------------------------------------------
# AI reasoning section (TA-E5-S1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_reasoning_section_shows_placeholder_by_default():
    """AI reasoning section shows placeholder when no reasoning is loaded."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    text = panel._reasoning_text.toPlainText()
    assert "click Analyze" in text or "No reasoning" in text


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_load_reasoning_displays_content():
    """load_reasoning(data) populates the reasoning section with formatted text."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    panel.load_reasoning({
        "setup_type": "Type A — EP",
        "profit_mechanism": "PM-02 PEAD",
        "thesis": "Strong earnings beat with institutional follow-through",
        "entry": 186.50,
        "stop": 173.50,
        "target": 212.00,
        "confidence": "HIGH",
    })
    text = panel._reasoning_text.toPlainText()
    assert "Type A" in text
    assert "PM-02" in text
    assert "186.50" in text


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_load_reasoning_none_resets_to_placeholder():
    """load_reasoning(None) resets to placeholder text."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    panel.load_reasoning({"setup_type": "Type B", "profit_mechanism": "PM-01",
                           "thesis": "VCP breakout", "entry": 100.0, "stop": 93.0,
                           "target": 120.0, "confidence": "MEDIUM"})
    panel.load_reasoning(None)
    text = panel._reasoning_text.toPlainText()
    assert "click Analyze" in text or "No reasoning" in text


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_load_reasoning_short_side_rr_is_positive():
    """R:R must be positive for short trades where stop > entry."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    # Short: entry 50, stop 57 (above entry), target 35 (below entry)
    panel.load_reasoning({
        "setup_type": "Type D",
        "profit_mechanism": "PM-05",
        "thesis": "RW breakdown",
        "entry": 50.0,
        "stop": 57.0,
        "target": 35.0,
        "confidence": "MEDIUM",
    })
    text = panel._reasoning_text.toPlainText()
    # R:R = abs(35-50)/abs(57-50) = 15/7 ≈ 2.1
    assert "2." in text  # R:R should be ~2.1:1, not negative


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_load_row_clears_reasoning():
    """Selecting a new row clears any previously loaded reasoning."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    panel.load_reasoning({"setup_type": "Type A", "profit_mechanism": "PM-02",
                           "thesis": "EP trade", "entry": 186.0, "stop": 173.0,
                           "target": 210.0, "confidence": "HIGH"})
    # Load a new row — reasoning should reset
    panel.load_row(_FULL_ROW)
    text = panel._reasoning_text.toPlainText()
    assert "click Analyze" in text or "No reasoning" in text


# ---------------------------------------------------------------------------
# Analyze button (TA-E5-S2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_detail_panel_has_analyze_button():
    """Panel must expose an _analyze_btn attribute."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    assert hasattr(panel, "_analyze_btn")


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_analyze_button_emits_analyze_requested_signal():
    """Clicking Analyze emits analyze_requested with the current row dict."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)

    emitted: list[dict] = []
    panel.analyze_requested.connect(emitted.append)
    panel._analyze_btn.click()

    assert len(emitted) == 1
    assert emitted[0]["symbol"] == "AAPL"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_set_analyzing_true_disables_analyze_button():
    """set_analyzing(True) disables the Analyze button."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    panel.set_analyzing(True)
    assert not panel._analyze_btn.isEnabled()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_set_analyzing_false_reenables_analyze_button():
    """set_analyzing(False) re-enables the Analyze button."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._detail_panel import CandidateDetailPanel

    ensure_qt_app()
    panel = CandidateDetailPanel()
    panel.load_row(_FULL_ROW)
    panel.set_analyzing(True)
    panel.set_analyzing(False)
    assert panel._analyze_btn.isEnabled()
