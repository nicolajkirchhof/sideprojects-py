"""
finance.apps.conditions
========================
Market conditions dashboard — GO/NO-GO regime panel + DRIFT eligibility.

Launch::

    python -m finance.apps conditions
"""
from __future__ import annotations

APP_NAME = "conditions"
APP_DESCRIPTION = "Market conditions dashboard (regime + DRIFT eligibility)"
APP_ICON_ID = "conditions"

_GLOBAL_WIN = None


def launch(**_kwargs) -> None:
    """Launch (or re-use) the conditions dashboard."""
    global _GLOBAL_WIN

    from finance.apps._qt_bootstrap import (
        ensure_qt_app,
        ensure_ipython_event_loop,
        exec_or_return,
    )
    from ._window import ConditionsWindow

    ensure_ipython_event_loop()
    app = ensure_qt_app()

    if _GLOBAL_WIN is None:
        _GLOBAL_WIN = ConditionsWindow()

    _GLOBAL_WIN.show()
    exec_or_return(app)
