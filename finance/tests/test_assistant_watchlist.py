"""
Tests for finance.apps.assistant._watchlist_model — TA-E4-S1 / TA-E4-S3.

Tests are written first (TDD). Qt model tests don't require a display
(no rendering), but they do need a QApplication for QColor/QBrush.
ensure_qt_app() creates it once at import time.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_ROWS = [
    {
        "symbol": "AAPL",
        "direction": "long",
        "price": 185.0,
        "change_5d_pct": 5.2,
        "rvol_20d": 1.8,
        "sector": "Technology",
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
    },
    {
        "symbol": "TSLA",
        "direction": "short",
        "price": 250.0,
        "change_5d_pct": -3.1,
        "rvol_20d": 2.5,
        "sector": "Consumer",
        "score_total": 35.0,
        "score_tag_bonus": 2.0,
        "tags": ["pead-short"],
        "dimensions": [
            {"dimension": 1, "weighted_score": 8.0},
            {"dimension": 2, "weighted_score": 7.0},
            {"dimension": 3, "weighted_score": 5.0},
            {"dimension": 4, "weighted_score": 10.0},
            {"dimension": 5, "weighted_score": 5.0},
        ],
    },
    {
        "symbol": "MSFT",
        "direction": "long",
        "price": 420.0,
        "change_5d_pct": 1.0,
        "rvol_20d": 0.9,
        "sector": "Technology",
        "score_total": 55.0,
        "score_tag_bonus": 0.0,
        "tags": [],
        "dimensions": [
            {"dimension": 1, "weighted_score": 15.0},
            {"dimension": 2, "weighted_score": 12.0},
            {"dimension": 3, "weighted_score": 10.0},
            {"dimension": 4, "weighted_score": 12.0},
            {"dimension": 5, "weighted_score": 6.0},
        ],
    },
]

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)


def _model_with_rows():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._watchlist_model import WatchlistModel

    ensure_qt_app()
    m = WatchlistModel()
    m.load_rows(_TEST_ROWS)
    return m


# ---------------------------------------------------------------------------
# Row / column count
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_model_empty_initially():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._watchlist_model import WatchlistModel

    ensure_qt_app()
    m = WatchlistModel()
    assert m.rowCount() == 0


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_model_row_count_after_load():
    m = _model_with_rows()
    assert m.rowCount() == 3


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_model_column_count():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._watchlist_model import WatchlistModel, COLUMN_COUNT

    ensure_qt_app()
    m = WatchlistModel()
    assert m.columnCount() == COLUMN_COUNT


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_load_rows_replaces_existing():
    m = _model_with_rows()
    m.load_rows([_TEST_ROWS[0]])
    assert m.rowCount() == 1


# ---------------------------------------------------------------------------
# Display values
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_symbol_display():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    idx = m.index(0, Col.SYMBOL)
    assert m.data(idx, QtCore.Qt.ItemDataRole.DisplayRole) == "AAPL"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_direction_display_abbreviated():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    long_idx = m.index(0, Col.DIRECTION)
    short_idx = m.index(1, Col.DIRECTION)
    assert m.data(long_idx, QtCore.Qt.ItemDataRole.DisplayRole) == "L"
    assert m.data(short_idx, QtCore.Qt.ItemDataRole.DisplayRole) == "S"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_score_display_one_decimal():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    idx = m.index(0, Col.SCORE)
    assert m.data(idx, QtCore.Qt.ItemDataRole.DisplayRole) == "75.0"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_d1_score_display():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    idx = m.index(0, Col.D1)
    assert m.data(idx, QtCore.Qt.ItemDataRole.DisplayRole) == "20.0"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_tags_joined_as_string():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    idx = m.index(0, Col.TAGS)
    assert m.data(idx, QtCore.Qt.ItemDataRole.DisplayRole) == "52w-high, vol-spike"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_change_5d_sign_and_percent():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    pos_idx = m.index(0, Col.CHANGE_5D)
    neg_idx = m.index(1, Col.CHANGE_5D)
    assert m.data(pos_idx, QtCore.Qt.ItemDataRole.DisplayRole) == "+5.2%"
    assert m.data(neg_idx, QtCore.Qt.ItemDataRole.DisplayRole) == "-3.1%"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_rvol_display_with_x_suffix():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    idx = m.index(0, Col.RVOL)
    assert m.data(idx, QtCore.Qt.ItemDataRole.DisplayRole) == "1.8x"


# ---------------------------------------------------------------------------
# Score colour coding
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_score_background_green_above_70():
    from pyqtgraph.Qt import QtCore, QtGui
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col, _COLOR_GREEN

    m = _model_with_rows()
    idx = m.index(0, Col.SCORE)  # AAPL score=75
    brush = m.data(idx, QtCore.Qt.ItemDataRole.BackgroundRole)
    assert isinstance(brush, QtGui.QBrush)
    assert brush.color() == QtGui.QColor(_COLOR_GREEN)


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_score_background_amber_between_40_and_70():
    from pyqtgraph.Qt import QtCore, QtGui
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col, _COLOR_AMBER

    m = _model_with_rows()
    idx = m.index(2, Col.SCORE)  # MSFT score=55
    brush = m.data(idx, QtCore.Qt.ItemDataRole.BackgroundRole)
    assert isinstance(brush, QtGui.QBrush)
    assert brush.color() == QtGui.QColor(_COLOR_AMBER)


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_score_background_red_below_40():
    from pyqtgraph.Qt import QtCore, QtGui
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col, _COLOR_RED

    m = _model_with_rows()
    idx = m.index(1, Col.SCORE)  # TSLA score=35
    brush = m.data(idx, QtCore.Qt.ItemDataRole.BackgroundRole)
    assert isinstance(brush, QtGui.QBrush)
    assert brush.color() == QtGui.QColor(_COLOR_RED)


# ---------------------------------------------------------------------------
# Checkbox state
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_checkbox_initially_unchecked():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    idx = m.index(0, Col.CHECK)
    state = m.data(idx, QtCore.Qt.ItemDataRole.CheckStateRole)
    assert state == QtCore.Qt.CheckState.Unchecked


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_checkbox_toggle_checked():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    idx = m.index(0, Col.CHECK)
    m.setData(idx, QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
    state = m.data(idx, QtCore.Qt.ItemDataRole.CheckStateRole)
    assert state == QtCore.Qt.CheckState.Checked


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_checked_symbols_returns_selected():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    m.setData(m.index(0, Col.CHECK), QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
    m.setData(m.index(2, Col.CHECK), QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
    syms = m.checked_symbols()
    assert "AAPL" in syms
    assert "MSFT" in syms
    assert "TSLA" not in syms


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_load_rows_resets_checkboxes():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    m.setData(m.index(0, Col.CHECK), QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
    m.load_rows(_TEST_ROWS)  # reload
    state = m.data(m.index(0, Col.CHECK), QtCore.Qt.ItemDataRole.CheckStateRole)
    assert state == QtCore.Qt.CheckState.Unchecked


# ---------------------------------------------------------------------------
# Sort values (UserRole)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_sort_value_score_is_float():
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    v = m.data(m.index(0, Col.SCORE), QtCore.Qt.ItemDataRole.UserRole)
    assert isinstance(v, float)
    assert v == pytest.approx(75.0)


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_sort_value_missing_numeric_is_sentinel():
    from pyqtgraph.Qt import QtCore
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col, _MISSING_SORT_VALUE

    ensure_qt_app()
    m = WatchlistModel()
    m.load_rows([{"symbol": "X", "direction": "long", "score_total": 50.0,
                  "price": None, "tags": [], "dimensions": []}])
    v = m.data(m.index(0, Col.PRICE), QtCore.Qt.ItemDataRole.UserRole)
    assert v == _MISSING_SORT_VALUE


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_data_out_of_range_column_returns_none():
    """data() must return None for a column index outside the Col enum."""
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, COLUMN_COUNT

    m = _model_with_rows()
    # Use an index one past the last valid column
    idx = m.index(0, COLUMN_COUNT)
    result = m.data(idx, QtCore.Qt.ItemDataRole.DisplayRole)
    assert result is None


# ---------------------------------------------------------------------------
# Window integration — centre panel is a QTableView, not a placeholder
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_centre_panel_is_table():
    from pyqtgraph.Qt import QtWidgets
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._window import AssistantWindow

    ensure_qt_app()
    apply_dark_palette(ensure_qt_app())
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=None):
        win = AssistantWindow()
    # Centre panel should contain a QTableView
    table = win._centre_panel.findChild(QtWidgets.QTableView)
    assert table is not None
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_watchlist_model_loaded_from_cache():
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._window import AssistantWindow

    ensure_qt_app()
    apply_dark_palette(ensure_qt_app())
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=_TEST_ROWS):
        win = AssistantWindow()
    assert win._watchlist_model.rowCount() == 3
    win.close()


# ---------------------------------------------------------------------------
# Batch selection — TA-E4-S3
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_check_top_n_checks_highest_scored():
    """check_top_n(1) should check only the row with the highest score_total."""
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    count = m.check_top_n(1)
    assert count == 1
    # AAPL has the highest score (75.0) and is at source index 0
    assert "AAPL" in m.checked_symbols()
    assert "TSLA" not in m.checked_symbols()
    assert "MSFT" not in m.checked_symbols()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_check_top_n_clamps_to_row_count():
    """check_top_n(n > rowCount) should check all rows and return rowCount."""
    m = _model_with_rows()
    count = m.check_top_n(100)
    assert count == 3
    assert set(m.checked_symbols()) == {"AAPL", "TSLA", "MSFT"}


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_uncheck_all_clears_checked():
    """uncheck_all() should clear all checked rows."""
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    m.setData(m.index(0, Col.CHECK), QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
    m.setData(m.index(1, Col.CHECK), QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
    assert m.checked_count() == 2
    m.uncheck_all()
    assert m.checked_count() == 0
    assert m.checked_symbols() == []


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_checked_count_reflects_checked_rows():
    """checked_count() should return the number of currently checked rows."""
    from pyqtgraph.Qt import QtCore
    from finance.apps.assistant._watchlist_model import WatchlistModel, Col

    m = _model_with_rows()
    assert m.checked_count() == 0
    m.setData(m.index(0, Col.CHECK), QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
    assert m.checked_count() == 1
    m.setData(m.index(2, Col.CHECK), QtCore.Qt.CheckState.Checked, QtCore.Qt.ItemDataRole.CheckStateRole)
    assert m.checked_count() == 2


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_check_rows_checks_specific_source_indices():
    """check_rows([0, 2]) should check AAPL and MSFT but not TSLA."""
    m = _model_with_rows()
    m.check_rows([0, 2])
    assert "AAPL" in m.checked_symbols()
    assert "MSFT" in m.checked_symbols()
    assert "TSLA" not in m.checked_symbols()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_left_panel_is_swing_regime_panel():
    """Left panel must be a SwingRegimePanel, not a generic placeholder."""
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._swing_panel import SwingRegimePanel
    from finance.apps.assistant._window import AssistantWindow

    ensure_qt_app()
    apply_dark_palette(ensure_qt_app())
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=None), \
         patch("finance.apps.assistant._window.load_daily", return_value=None):
        win = AssistantWindow()
    assert isinstance(win._left_panel, SwingRegimePanel)
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_export_buttons_enabled_after_check():
    """Export buttons should become enabled once at least one row is checked."""
    from pyqtgraph.Qt import QtCore
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._window import AssistantWindow

    ensure_qt_app()
    apply_dark_palette(ensure_qt_app())
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=_TEST_ROWS):
        win = AssistantWindow()

    assert not win._btn_export_bc.isEnabled()
    assert not win._btn_export_tws.isEnabled()

    # Check one row via the model — window must respond
    win._watchlist_model.check_rows([0])
    assert win._btn_export_bc.isEnabled()
    assert win._btn_export_tws.isEnabled()
    win.close()
