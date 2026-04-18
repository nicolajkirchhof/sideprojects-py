"""
finance.apps._launcher
========================
Qt launcher window — grid of icon buttons for launching finance apps.
"""
from __future__ import annotations

import importlib
import subprocess
import sys

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from finance.apps import APPS
from finance.apps._launcher_icons import icon_for_app

_BUTTON_ICON_SIZE = 64
_BUTTON_SIZE = 88


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Finance Apps")
        self.setFixedSize(self._calc_width(), 140)
        self._build_ui()
        self._center_on_screen()

    def _calc_width(self) -> int:
        n = max(len(APPS), 1)
        return n * (_BUTTON_SIZE + 12) + 24

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        for name, module_path in APPS.items():
            mod = importlib.import_module(module_path)
            description = getattr(mod, "APP_DESCRIPTION", name)
            icon_id = getattr(mod, "APP_ICON_ID", None)

            col = QtWidgets.QVBoxLayout()
            col.setSpacing(4)

            btn = QtWidgets.QToolButton()
            btn.setIcon(icon_for_app(icon_id, name))
            btn.setIconSize(QtCore.QSize(_BUTTON_ICON_SIZE, _BUTTON_ICON_SIZE))
            btn.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
            btn.setToolTip(description)
            btn.setToolButtonStyle(
                QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
            )
            btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            btn.clicked.connect(lambda _checked, n=name: self._launch_app(n))
            col.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

            label = QtWidgets.QLabel(name)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size: 11px;")
            col.addWidget(label)

            layout.addLayout(col)

    def _center_on_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )

    @staticmethod
    def _launch_app(name: str) -> None:
        mod = importlib.import_module(APPS[name])
        is_gui = getattr(mod, "APP_GUI", True)

        kwargs = {}
        if not is_gui and sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

        subprocess.Popen(
            [sys.executable, "-m", "finance.apps", name],
            start_new_session=is_gui,
            **kwargs,
        )


def launch_launcher() -> None:
    """Entry point for the launcher window."""
    from finance.apps._qt_bootstrap import (
        ensure_qt_app,
        apply_dark_palette,
        exec_or_return,
    )

    app = ensure_qt_app()
    apply_dark_palette(app)

    win = LauncherWindow()
    win.show()
    exec_or_return(app, force_exec=True)
