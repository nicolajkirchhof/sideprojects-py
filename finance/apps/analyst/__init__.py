"""
finance.apps.analyst
=====================
Daily trade analyst pipeline — scanner ingestion, 5-box scoring,
Claude-powered analysis, and Tradelog integration.

Usage::

    python -m finance.apps analyst          # run full pipeline
    python -m finance.apps analyst --dry-run  # score only, no Claude/push
"""
from __future__ import annotations

APP_NAME = "analyst"
APP_DESCRIPTION = "Daily trade analyst (scanner → 5-box → Claude → Tradelog)"
APP_ICON_ID = "analyst"
APP_GUI = False  # non-GUI pipeline — launcher opens a visible console


def launch(**kwargs) -> None:  # noqa: ANN003
    """Entry point called by the app launcher."""
    from finance.apps.analyst._pipeline import run
    run(**kwargs)
