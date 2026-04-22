"""
Tests for finance.apps.assistant window shell — TA-E2-S1.

Tests are written first (TDD). They define expected behaviour before
the implementation exists.

Qt window instantiation requires a display. The smoke test is guarded
with skipif so it passes in headless CI while still running locally.
"""
from __future__ import annotations

import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Metadata tests — no Qt needed
# ---------------------------------------------------------------------------


def test_app_name():
    from finance.apps.assistant import APP_NAME
    assert APP_NAME == "assistant"


def test_app_description():
    from finance.apps.assistant import APP_DESCRIPTION
    assert isinstance(APP_DESCRIPTION, str)
    assert len(APP_DESCRIPTION) > 0


def test_app_registered_in_registry():
    from finance.apps import APPS
    assert "assistant" in APPS


def test_conditions_removed_from_registry():
    from finance.apps import APPS
    assert "conditions" not in APPS


def test_analyst_removed_from_registry():
    from finance.apps import APPS
    assert "analyst" not in APPS


def test_launch_function_exists():
    from finance.apps.assistant import launch
    assert callable(launch)


# ---------------------------------------------------------------------------
# Window import test — verifies module is importable, no Qt instantiation
# ---------------------------------------------------------------------------


def test_window_module_importable():
    from finance.apps.assistant import _window  # noqa: F401
    assert hasattr(_window, "AssistantWindow")


# ---------------------------------------------------------------------------
# Window smoke test — requires display
# ---------------------------------------------------------------------------

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_instantiates():
    """AssistantWindow can be created without crashing."""
    from finance.apps._qt_bootstrap import ensure_qt_app, apply_dark_palette
    from finance.apps.assistant._window import AssistantWindow

    app = ensure_qt_app()
    apply_dark_palette(app)
    win = AssistantWindow()

    assert win.windowTitle() == "Trading Assistant"
    assert win.size().width() > 0
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_has_three_panels():
    """AssistantWindow exposes left, centre, and right panel widgets."""
    from finance.apps._qt_bootstrap import ensure_qt_app, apply_dark_palette
    from finance.apps.assistant._window import AssistantWindow

    app = ensure_qt_app()
    apply_dark_palette(app)
    win = AssistantWindow()

    assert hasattr(win, "_left_panel")
    assert hasattr(win, "_centre_panel")
    assert hasattr(win, "_right_panel")
    win.close()


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_window_has_run_pipeline_button():
    """AssistantWindow exposes the Run Pipeline toolbar button."""
    from finance.apps._qt_bootstrap import ensure_qt_app, apply_dark_palette
    from finance.apps.assistant._window import AssistantWindow

    app = ensure_qt_app()
    apply_dark_palette(app)
    win = AssistantWindow()

    assert hasattr(win, "_btn_run")
    win.close()
