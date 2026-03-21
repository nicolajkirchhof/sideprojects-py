"""
finance.apps.momentum_dashboard
=================================
Momentum & Earnings analysis dashboard (PySide6/PyQtGraph).

Launch::

    python -m finance.apps momentum
"""
from __future__ import annotations

APP_NAME = "momentum"
APP_DESCRIPTION = "Momentum & Earnings analysis dashboard (Qt)"


_GLOBAL_WIN = None
_GLOBAL_LOADED_DF = None


def launch(start_year: int = 2022, **_kwargs) -> None:
    """Launch (or re-use) the momentum dashboard."""
    global _GLOBAL_WIN, _GLOBAL_LOADED_DF

    import pandas as pd

    from finance.apps._qt_bootstrap import (
        ensure_qt_app, apply_dark_palette,
        ensure_ipython_event_loop, exec_or_return,
    )
    from finance.utils.momentum_data import load_and_prep_data
    from ._dashboard import DashboardQt

    ensure_ipython_event_loop()
    app = ensure_qt_app()
    apply_dark_palette(app)

    # Reuse existing window if visible
    if _GLOBAL_WIN is not None:
        try:
            if not _GLOBAL_WIN.isVisible():
                _GLOBAL_WIN = None
        except Exception:
            _GLOBAL_WIN = None

    # Load data
    if _GLOBAL_LOADED_DF is not None and not _GLOBAL_LOADED_DF.empty:
        df = _GLOBAL_LOADED_DF
    else:
        this_year = int(pd.Timestamp.now().year)
        df = load_and_prep_data(range(start_year, this_year + 1))
        _GLOBAL_LOADED_DF = df

    if _GLOBAL_WIN is None:
        _GLOBAL_WIN = DashboardQt(df)
    else:
        _GLOBAL_WIN.df = df
        _GLOBAL_LOADED_DF = df
        _GLOBAL_WIN._update_cond_label_and_bounds()
        _GLOBAL_WIN._refresh_filter_bounds_from_df()
        _GLOBAL_WIN._show_empty_state()
        _GLOBAL_WIN._update_data_status()

    _GLOBAL_WIN.show()
    _GLOBAL_WIN.showMaximized()
    _GLOBAL_WIN.raise_()
    _GLOBAL_WIN.activateWindow()

    try:
        app.processEvents()
        app.processEvents()
    except Exception:
        pass

    exec_or_return(app)
