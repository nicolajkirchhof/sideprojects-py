from __future__ import annotations

import sys
from typing import Optional

# Prefer PySide6, fallback to PyQt5
try:
    from PySide6 import QtCore, QtWidgets
    QT_API = "PySide6"
except Exception:
    from PyQt5 import QtCore, QtWidgets  # type: ignore
    QT_API = "PyQt5"

_LAST_WIN: Optional[QtWidgets.QMainWindow] = None


class SmokeWindow(QtWidgets.QMainWindow):
    def __init__(self, with_pyqtgraph: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(f"Qt Smoke Test ({QT_API})")
        self.resize(900, 600)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.lbl = QtWidgets.QLabel("If this text updates every 250ms, the Qt event loop is healthy.")
        self.lbl.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.lbl)

        row = QtWidgets.QHBoxLayout()
        layout.addLayout(row)

        self.btn = QtWidgets.QPushButton("Click me")
        self.btn.clicked.connect(self._clicked)
        row.addWidget(self.btn)

        self.chk = QtWidgets.QCheckBox("Enable pyqtgraph widget")
        self.chk.setChecked(with_pyqtgraph)
        self.chk.toggled.connect(self._toggle_pg)
        row.addWidget(self.chk)

        row.addStretch(1)

        # Placeholder where pyqtgraph will be inserted
        self.pg_container = QtWidgets.QWidget()
        self.pg_layout = QtWidgets.QVBoxLayout(self.pg_container)
        self.pg_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pg_container, 1)

        self._tick = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(250)

        self._pg_widget = None
        if with_pyqtgraph:
            self._create_pyqtgraph()

    def _clicked(self) -> None:
        self.lbl.setText(f"Button clicked. Still alive. Tick={self._tick}")

    def _on_tick(self) -> None:
        self._tick += 1
        self.lbl.setText(f"Tick={self._tick}  |  Qt API={QT_API}")

    def _toggle_pg(self, on: bool) -> None:
        if on and self._pg_widget is None:
            self._create_pyqtgraph()
        elif (not on) and self._pg_widget is not None:
            self._destroy_pyqtgraph()

    def _create_pyqtgraph(self) -> None:
        # Import only when needed so you can test "Qt only" vs "Qt+pyqtgraph"
        import numpy as np
        import pyqtgraph as pg

        pg.setConfigOptions(antialias=True)

        pw = pg.PlotWidget()
        pw.setMinimumHeight(300)
        pw.setBackground("k")
        pw.showGrid(x=True, y=True, alpha=0.2)
        pw.setTitle("pyqtgraph plot: if you see this, graphics stack is OK", color="#DDDDDD", size="12pt")

        x = np.linspace(0, 6.28, 400)
        y = np.sin(x)
        pw.plot(x, y, pen=pg.mkPen("c", width=2))

        self.pg_layout.addWidget(pw)
        self._pg_widget = pw

    def _destroy_pyqtgraph(self) -> None:
        w = self._pg_widget
        self._pg_widget = None
        if w is not None:
            w.setParent(None)
            w.deleteLater()


def show_window(*, with_pyqtgraph: bool = True, exec_: Optional[bool] = None) -> QtWidgets.QMainWindow:
    """
    Shows the smoke test window.

    - In a normal Python run: call show_window(exec_=True) or run this file directly.
    - In IPython: do `%gui qt` and call show_window(exec_=False) (default usually fine).
    """
    global _LAST_WIN

    app = QtWidgets.QApplication.instance()
    created = False
    if app is None:
        # Use sys.argv[:1] to avoid passing IPython args into Qt
        app = QtWidgets.QApplication(sys.argv[:1])
        created = True

    win = SmokeWindow(with_pyqtgraph=with_pyqtgraph)
    win.show()

    # keep strong reference in interactive sessions
    _LAST_WIN = win

    if exec_ is None:
        exec_ = created  # if we created the app, we should run the loop

    if exec_:
        app.exec()

    return win


if __name__ == "__main__":
    show_window(with_pyqtgraph=True, exec_=True)
