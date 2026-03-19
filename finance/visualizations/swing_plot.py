"""
finance.visualizations.swing_plot
===================================
Entry point for the swing trading analysis dashboard.

Run directly:
    python finance/visualizations/swing_plot.py

Or call interactive() from an IPython session.

Module layout
-------------
_config.py  — plot config constants (re-exported here for backward compat)
_items.py   — custom PyQtGraph graphics items
_chart.py   — chart pane setup, data binding, auto-scaling
_tabs.py    — matplotlib tab renderers
_app.py     — SwingPlotWindow (QMainWindow subclass)
"""
import sys
import os

# Ensure project root is on sys.path so the package is importable when
# this file is run directly as a script (python finance/visualizations/swing_plot.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyqtgraph as pg
from pyqtgraph import QtCore
from pyqtgraph.Qt import QtWidgets

# Re-export config constants so existing code that does
# `from finance.visualizations.swing_plot import MA_CONFIGS` keeps working.
from finance.visualizations._config import (  # noqa: F401
    MA_CONFIGS, ATR_CONFIGS, SLOPE_CONFIGS, VOL_CONFIGS, DIST_CONFIGS,
    HV_CONFIGS, IVPCT_CONFIGS, BB_CONFIGS, TTM_COLORS,
    ATR_RATIO_THRESHOLD, ATR_RATIO_COLOR,
)
from finance.visualizations._app import SwingPlotWindow


_GLOBAL_QT_APP: QtWidgets.QApplication | None = None
_GLOBAL_WIN:    SwingPlotWindow | None         = None


def interactive(default_symbol: str = 'SPY', datasource: str = 'offline'):
    """
    Launch (or re-use) the swing plot dashboard.

    The Qt application and main window are cached as module-level singletons
    so that calling interactive() a second time from an IPython session simply
    refreshes the existing window rather than opening a new one.
    """
    global _GLOBAL_QT_APP, _GLOBAL_WIN

    if _GLOBAL_QT_APP is None:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        _GLOBAL_QT_APP = pg.mkQApp()

    if _GLOBAL_WIN is None:
        _GLOBAL_WIN = SwingPlotWindow()

    _GLOBAL_WIN.load_symbol(default_symbol, datasource)
    _GLOBAL_WIN.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()


if __name__ == '__main__':
    interactive()
