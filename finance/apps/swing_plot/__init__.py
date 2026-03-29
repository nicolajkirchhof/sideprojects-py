"""
finance.apps.swing_plot
========================
Swing trading analysis dashboard (PyQtGraph).

Launch::

    python -m finance.apps swing-plot
"""
from __future__ import annotations

APP_NAME = "swing-plot"
APP_DESCRIPTION = "Swing trading analysis dashboard (multi-tab PyQtGraph)"
APP_ICON_ID = "candlestick"


_GLOBAL_WIN = None


def launch(default_symbol: str = "SPY", datasource: str = "offline", **_kwargs) -> None:
    """Launch (or re-use) the swing plot dashboard."""
    global _GLOBAL_WIN

    from finance.apps._qt_bootstrap import (
        ensure_qt_app, ensure_ipython_event_loop, exec_or_return,
    )
    from ._app import SwingPlotWindow

    ensure_ipython_event_loop()
    app = ensure_qt_app()

    if _GLOBAL_WIN is None:
        _GLOBAL_WIN = SwingPlotWindow()

    _GLOBAL_WIN.load_symbol(default_symbol, datasource)
    _GLOBAL_WIN.show()

    exec_or_return(app)
