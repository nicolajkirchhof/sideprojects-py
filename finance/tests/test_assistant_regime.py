"""
Tests for TA-E3-S2 (stop-out counter + GO/NO-GO override)
and TA-E3-S3 (economic events calendar cache I/O).

Written test-first (TDD).
"""
from __future__ import annotations

import json
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

_SAMPLE_EVENTS = [
    {
        "title": "Non-Farm Payrolls",
        "country": "USD",
        "date": "2026-04-25T12:30:00+00:00",
        "impact": "High",
        "forecast": "200K",
        "previous": "185K",
    },
    {
        "title": "CPI m/m",
        "country": "USD",
        "date": "2026-04-28T12:30:00+00:00",
        "impact": "High",
        "forecast": "0.3%",
        "previous": "0.2%",
    },
]


# ---------------------------------------------------------------------------
# StopOutCounter — model-level (no display required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_stopout_counter_starts_at_zero():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import StopOutCounter

    ensure_qt_app()
    counter = StopOutCounter()
    assert counter.count == 0


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_stopout_counter_increments():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import StopOutCounter

    ensure_qt_app()
    counter = StopOutCounter()
    counter.increment()
    counter.increment()
    assert counter.count == 2


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_stopout_counter_clamps_at_zero():
    """Decrementing below zero should keep count at 0."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import StopOutCounter

    ensure_qt_app()
    counter = StopOutCounter()
    counter.decrement()
    counter.decrement()
    assert counter.count == 0


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_stopout_counter_three_triggers_override():
    """At count == 3 the override should be active."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import StopOutCounter

    ensure_qt_app()
    counter = StopOutCounter()
    assert not counter.is_override_active
    counter.increment()
    counter.increment()
    assert not counter.is_override_active
    counter.increment()
    assert counter.is_override_active


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_stopout_reset_clears_override():
    """Resetting the counter should deactivate the override."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._swing_panel import StopOutCounter

    ensure_qt_app()
    counter = StopOutCounter()
    counter.increment()
    counter.increment()
    counter.increment()
    assert counter.is_override_active
    counter.reset()
    assert counter.count == 0
    assert not counter.is_override_active


# ---------------------------------------------------------------------------
# Calendar cache I/O (no Qt required)
# ---------------------------------------------------------------------------


def test_write_cache_includes_events(tmp_path):
    from finance.apps.assistant._pipeline import write_cache

    rows = [{"symbol": "AAPL", "score_total": 72.5}]
    path = write_cache(rows, _TEST_DATE, base_dir=tmp_path, events=_SAMPLE_EVENTS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "events" in payload
    assert len(payload["events"]) == 2
    assert payload["events"][0]["title"] == "Non-Farm Payrolls"


def test_write_cache_events_defaults_to_empty(tmp_path):
    from finance.apps.assistant._pipeline import write_cache

    rows = [{"symbol": "AAPL", "score_total": 72.5}]
    path = write_cache(rows, _TEST_DATE, base_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["events"] == []


def test_read_events_from_cache_returns_events(tmp_path):
    from finance.apps.assistant._pipeline import read_events_from_cache, write_cache

    rows = [{"symbol": "AAPL", "score_total": 72.5}]
    write_cache(rows, _TEST_DATE, base_dir=tmp_path, events=_SAMPLE_EVENTS)
    events = read_events_from_cache(_TEST_DATE, base_dir=tmp_path)
    assert events is not None
    assert len(events) == 2
    assert events[1]["title"] == "CPI m/m"


def test_read_events_from_cache_returns_none_when_missing(tmp_path):
    from finance.apps.assistant._pipeline import read_events_from_cache

    result = read_events_from_cache(_TEST_DATE, base_dir=tmp_path)
    assert result is None


def test_read_events_from_cache_returns_none_when_no_events_key(tmp_path):
    """Old cache files without 'events' key should return None, not error."""
    from finance.apps.assistant._pipeline import cache_path, read_events_from_cache

    # Write cache file without events key (old format)
    p = cache_path(_TEST_DATE, base_dir=tmp_path)
    p.write_text(
        json.dumps({"date": "2026-04-23", "rows": [], "created_at": "2026-04-23T18:00:00"}),
        encoding="utf-8",
    )
    result = read_events_from_cache(_TEST_DATE, base_dir=tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# Window integration (require display)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_left_panel_has_stopout_counter():
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._swing_panel import StopOutCounter
    from finance.apps.assistant._window import AssistantWindow

    ensure_qt_app()
    apply_dark_palette(ensure_qt_app())
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=None), \
         patch("finance.apps.assistant._window.load_daily", return_value=None):
        win = AssistantWindow()
    assert hasattr(win._left_panel, "stop_counter")
    assert isinstance(win._left_panel.stop_counter, StopOutCounter)
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_left_panel_has_events_widget():
    from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
    from finance.apps.assistant._swing_panel import EventsCalendarWidget
    from finance.apps.assistant._window import AssistantWindow

    ensure_qt_app()
    apply_dark_palette(ensure_qt_app())
    with patch("finance.apps.assistant._pipeline.read_cache", return_value=None), \
         patch("finance.apps.assistant._window.load_daily", return_value=None):
        win = AssistantWindow()
    assert hasattr(win._left_panel, "events_widget")
    assert isinstance(win._left_panel.events_widget, EventsCalendarWidget)
    win.close()
