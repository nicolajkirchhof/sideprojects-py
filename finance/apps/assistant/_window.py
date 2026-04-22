"""
finance.apps.assistant._window
================================
Main window for the Trading Assistant.

Three-panel layout (QSplitter):
  Left  (300px)  — Market Context panel
  Centre (flex)  — Watchlist Table panel
  Right (350px)  — Detail Panel

Window geometry and splitter positions are persisted across restarts
via QSettings (key prefix: TradingAssistant/).
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

_SETTINGS_ORG = "sideprojects-py"
_SETTINGS_APP = "TradingAssistant"
_KEY_GEOMETRY = "geometry"
_KEY_SPLITTER = "splitter"

_DEFAULT_WIDTH = 1400
_DEFAULT_HEIGHT = 900
_LEFT_WIDTH = 300
_RIGHT_WIDTH = 350


class AssistantWindow(QtWidgets.QMainWindow):
    """
    Trading Assistant main window.

    Hosts three resizable panels and a toolbar. Panel content is populated
    by subsequent stories; this class provides only the shell and layout.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Trading Assistant")
        self.resize(_DEFAULT_WIDTH, _DEFAULT_HEIGHT)

        self._settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)

        self._build_toolbar()
        self._build_panels()
        self._build_status_bar()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(20, 20))

        self._btn_run = QtWidgets.QToolButton()
        self._btn_run.setText("▶  Run Pipeline")
        self._btn_run.setToolTip("Fetch scanner emails, enrich, score, and analyse candidates")
        self._btn_run.setMinimumWidth(130)
        tb.addWidget(self._btn_run)

        tb.addSeparator()

        self._btn_load_csv = QtWidgets.QToolButton()
        self._btn_load_csv.setText("Load CSV…")
        self._btn_load_csv.setToolTip("Manually import a Barchart screener CSV")
        tb.addWidget(self._btn_load_csv)

        tb.addSeparator()

        self._btn_export_bc = QtWidgets.QToolButton()
        self._btn_export_bc.setText("Export → Barchart")
        self._btn_export_bc.setToolTip("Copy selected tickers to clipboard (comma-separated)")
        self._btn_export_bc.setEnabled(False)
        tb.addWidget(self._btn_export_bc)

        self._btn_export_tws = QtWidgets.QToolButton()
        self._btn_export_tws.setText("Export → TWS")
        self._btn_export_tws.setToolTip("Save selected tickers as a TWS-importable CSV")
        self._btn_export_tws.setEnabled(False)
        tb.addWidget(self._btn_export_tws)

        # Push the rest to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        tb.addWidget(spacer)

        self._lbl_last_run = QtWidgets.QLabel("")
        self._lbl_last_run.setStyleSheet("color: #666; font-size: 11px; padding-right: 8px;")
        tb.addWidget(self._lbl_last_run)

    def _build_panels(self) -> None:
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self._left_panel = self._make_placeholder_panel("Market Context")
        self._centre_panel = self._make_placeholder_panel("Watchlist")
        self._right_panel = self._make_placeholder_panel("Detail")

        self._splitter.addWidget(self._left_panel)
        self._splitter.addWidget(self._centre_panel)
        self._splitter.addWidget(self._right_panel)

        # Initial width distribution — centre gets all remaining space
        total = _DEFAULT_WIDTH
        centre = total - _LEFT_WIDTH - _RIGHT_WIDTH
        self._splitter.setSizes([_LEFT_WIDTH, centre, _RIGHT_WIDTH])
        self._splitter.setStretchFactor(0, 0)  # left — fixed preference
        self._splitter.setStretchFactor(1, 1)  # centre — stretches
        self._splitter.setStretchFactor(2, 0)  # right — fixed preference

        self.setCentralWidget(self._splitter)

    def _build_status_bar(self) -> None:
        self._status_bar = self.statusBar()
        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_candidate_count = QtWidgets.QLabel("")
        self._status_bar.addWidget(self._lbl_status, 1)
        self._status_bar.addPermanentWidget(self._lbl_candidate_count)

    @staticmethod
    def _make_placeholder_panel(name: str) -> QtWidgets.QWidget:
        """Return a labelled placeholder widget for an unimplemented panel."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        lbl = QtWidgets.QLabel(name)
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #444; font-size: 16px;")
        layout.addWidget(lbl)

        return widget

    # ------------------------------------------------------------------
    # QSettings — geometry persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geom = self._settings.value(_KEY_GEOMETRY)
        if geom is not None:
            self.restoreGeometry(geom)

        splitter_state = self._settings.value(_KEY_SPLITTER)
        if splitter_state is not None:
            self._splitter.restoreState(splitter_state)

    def closeEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        self._settings.setValue(_KEY_GEOMETRY, self.saveGeometry())
        self._settings.setValue(_KEY_SPLITTER, self._splitter.saveState())
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Status bar helpers (used by pipeline thread in TA-E2-S2)
    # ------------------------------------------------------------------

    def set_status(self, message: str) -> None:
        """Update the left status bar label."""
        self._lbl_status.setText(message)

    def set_candidate_count(self, count: int) -> None:
        """Update the permanent candidate-count label."""
        self._lbl_candidate_count.setText(f"{count} candidates" if count is not None else "")
