"""
Tests for TA-E6-S1 (Barchart export) and TA-E6-S2 (TWS export).
Written test-first (TDD).
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)

_TEST_DATE = date(2026, 4, 23)
_SYMBOLS = ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# _export module — file I/O (no Qt required)
# ---------------------------------------------------------------------------


def test_export_barchart_creates_file(tmp_path):
    from finance.apps.assistant._export import export_barchart

    path = export_barchart(_SYMBOLS, _TEST_DATE, base_dir=tmp_path)
    assert path.exists()


def test_export_barchart_filename_format(tmp_path):
    from finance.apps.assistant._export import export_barchart

    path = export_barchart(_SYMBOLS, _TEST_DATE, base_dir=tmp_path)
    assert path.name == "watchlist-2026-04-23.txt"


def test_export_barchart_file_content(tmp_path):
    from finance.apps.assistant._export import export_barchart

    path = export_barchart(_SYMBOLS, _TEST_DATE, base_dir=tmp_path)
    content = path.read_text(encoding="utf-8").strip()
    assert content == "AAPL,MSFT,NVDA"


def test_export_barchart_empty_symbols(tmp_path):
    """Empty symbol list should write an empty file without error."""
    from finance.apps.assistant._export import export_barchart

    path = export_barchart([], _TEST_DATE, base_dir=tmp_path)
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() == ""


def test_export_tws_creates_file(tmp_path):
    from finance.apps.assistant._export import export_tws

    path = export_tws(_SYMBOLS, _TEST_DATE, base_dir=tmp_path)
    assert path.exists()


def test_export_tws_filename_format(tmp_path):
    from finance.apps.assistant._export import export_tws

    path = export_tws(_SYMBOLS, _TEST_DATE, base_dir=tmp_path)
    assert path.name == "tws-watchlist-2026-04-23.csv"


def test_export_tws_file_content(tmp_path):
    """Each line must be DES,SYMBOL,STK,SMART,,,, (all caps)."""
    from finance.apps.assistant._export import export_tws

    path = export_tws(_SYMBOLS, _TEST_DATE, base_dir=tmp_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[0] == "DES,AAPL,STK,SMART,,,,"
    assert lines[1] == "DES,MSFT,STK,SMART,,,,"
    assert lines[2] == "DES,NVDA,STK,SMART,,,,"


def test_export_tws_symbols_uppercased(tmp_path):
    """Symbols must be uppercased even if passed in lowercase."""
    from finance.apps.assistant._export import export_tws

    path = export_tws(["aapl", "msft"], _TEST_DATE, base_dir=tmp_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "DES,AAPL,STK,SMART,,,,"


def test_export_tws_empty_symbols(tmp_path):
    """Empty symbol list should write an empty file without error."""
    from finance.apps.assistant._export import export_tws

    path = export_tws([], _TEST_DATE, base_dir=tmp_path)
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() == ""


# ---------------------------------------------------------------------------
# Window integration (require display)
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
        "tags": ["52w-high"],
        "dimensions": [{"dimension": 1, "weighted_score": 20.0}],
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
        "dimensions": [{"dimension": 1, "weighted_score": 15.0}],
    },
]


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_export_barchart_button_copies_to_clipboard(tmp_path):
    """Clicking Export→Barchart copies checked symbols to clipboard."""
    from pyqtgraph.Qt import QtWidgets
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._window import AssistantWindow

    app = ensure_qt_app()
    apply_dark_palette(app)
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=_TEST_ROWS), \
         patch("finance.apps.assistant._window.load_daily", return_value=None):
        win = AssistantWindow()

    # Check one row then click export
    win._watchlist_model.check_rows([0])  # AAPL
    with patch("finance.apps.assistant._export._DEFAULT_EXPORT_DIR", tmp_path):
        win._btn_export_bc.click()

    text = QtWidgets.QApplication.clipboard().text()
    assert "AAPL" in text
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_export_tws_button_creates_file(tmp_path):
    """Clicking Export→TWS writes the TWS CSV file."""
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._window import AssistantWindow

    app = ensure_qt_app()
    apply_dark_palette(app)
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=_TEST_ROWS), \
         patch("finance.apps.assistant._window.load_daily", return_value=None):
        win = AssistantWindow()

    win._watchlist_model.check_rows([0, 1])
    with patch("finance.apps.assistant._export._DEFAULT_EXPORT_DIR", tmp_path), \
         patch("subprocess.Popen"):  # suppress Explorer
        win._btn_export_tws.click()

    files = list(tmp_path.glob("tws-watchlist-*.csv"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert any("AAPL" in line for line in lines)
    win.close()
