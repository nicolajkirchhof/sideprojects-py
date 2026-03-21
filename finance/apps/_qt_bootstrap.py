"""
finance.apps._qt_bootstrap
============================
Shared Qt application infrastructure for all finance apps.

Provides singleton QApplication management, dark palette, and
IPython event loop integration.
"""
from __future__ import annotations

import sys
from typing import Optional

import pyqtgraph as pg
from pyqtgraph import QtCore
from pyqtgraph.Qt import QtWidgets, QtGui


_QT_APP: Optional[QtWidgets.QApplication] = None
_IPYTHON_GUI_QT_ENABLED = False


def in_ipython() -> bool:
    """Best-effort detection of an IPython environment."""
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except Exception:
        return False


def ensure_ipython_event_loop() -> None:
    """In IPython/Jupyter, ensure the Qt event loop is integrated."""
    global _IPYTHON_GUI_QT_ENABLED
    if _IPYTHON_GUI_QT_ENABLED or not in_ipython():
        return
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("gui", "qt")
            _IPYTHON_GUI_QT_ENABLED = True
    except Exception:
        pass


def ensure_qt_app() -> QtWidgets.QApplication:
    """Create or return the singleton QApplication."""
    global _QT_APP
    if _QT_APP is None:
        QtCore.QCoreApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
        )
        _QT_APP = pg.mkQApp()
    return _QT_APP


def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    """Apply a dark Fusion palette. Safe to call repeatedly."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#111111"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#000000"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#DDDDDD"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#222222"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#DDDDDD"))
    app.setPalette(palette)


def exec_or_return(app: QtWidgets.QApplication, *, force_exec: Optional[bool] = None) -> None:
    """
    Run app.exec() if we are NOT in an interactive IPython session,
    or if force_exec is explicitly True.
    """
    if force_exec is None:
        force_exec = not in_ipython()
    if force_exec:
        app.exec()
