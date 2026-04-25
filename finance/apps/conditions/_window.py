"""Main window for the Conditions Dashboard."""
from __future__ import annotations

from datetime import datetime

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from finance.apps._qt_bootstrap import apply_dark_palette, ensure_qt_app
from finance.apps.conditions._data import (
    compute_go_nogo,
    compute_trend_status,
    compute_vix_status,
    load_daily,
)
from finance.apps.conditions._drift_panel import DriftPanel
from finance.apps.conditions._swing_panel import SwingRegimePanel


class ConditionsWindow(QtWidgets.QMainWindow):
    """Two-panel market conditions dashboard."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Market Conditions")
        self.resize(1200, 600)

        app = ensure_qt_app()
        apply_dark_palette(app)

        self._build_toolbar()
        self._build_panels()
        self._load_data()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(20, 20))

        self._btn_refresh = QtWidgets.QToolButton()
        self._btn_refresh.setText("\u27f3")  # ⟳
        self._btn_refresh.setToolTip("Refresh data (offline cache)")
        self._btn_refresh.clicked.connect(self._load_data)
        tb.addWidget(self._btn_refresh)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        tb.addWidget(spacer)

        self._lbl_updated = QtWidgets.QLabel("")
        self._lbl_updated.setStyleSheet("color: #666; font-size: 11px; padding-right: 8px;")
        tb.addWidget(self._lbl_updated)

    def _build_panels(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self._swing_panel = SwingRegimePanel()
        self._drift_panel = DriftPanel()

        splitter.addWidget(self._swing_panel)
        splitter.addWidget(self._drift_panel)
        splitter.setSizes([600, 600])

        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load cached parquet data and update all indicators."""
        df_spy = load_daily("SPY")
        df_qqq = load_daily("QQQ")
        df_vix = load_daily("$VIX")

        spy_trend = compute_trend_status("SPY", df_spy) if df_spy is not None else None
        qqq_trend = compute_trend_status("QQQ", df_qqq) if df_qqq is not None else None
        vix_status = compute_vix_status(df_vix) if df_vix is not None else None

        go_nogo = compute_go_nogo(spy_trend, vix_status)

        self._swing_panel.update_indicators(spy_trend, qqq_trend, vix_status, go_nogo)

        self._lbl_updated.setText(f"Updated {datetime.now():%H:%M:%S}")
