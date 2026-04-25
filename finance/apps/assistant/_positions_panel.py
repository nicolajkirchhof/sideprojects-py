"""
finance.apps.assistant._positions_panel
=========================================
Live Positions tab widget (TA-E7-S3).

Layout:
  Top-left  — QTableView with open positions + R-multiple + alert badge
  Top-right — AlertsPanel (scrollable rule alerts for selected row)
  Bottom    — Refresh button + last-updated timestamp

Auto-refreshes every 5 minutes while the window is open.
"""
from __future__ import annotations

import logging
from datetime import datetime

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

GREEN   = "#4CAF50"
AMBER   = "#FFA726"
RED     = "#F44336"
DIM     = "#666"

_SEVERITY_COLOURS = {"ok": GREEN, "warn": AMBER, "critical": RED}

# Auto-refresh interval (milliseconds)
_REFRESH_INTERVAL_MS = 5 * 60 * 1000


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------

_HEADERS = ["Symbol", "Type", "Dir", "Entry", "Current", "P&L $", "P&L R", "Days", "⚠"]
_COL_SYMBOL  = 0
_COL_TYPE    = 1
_COL_DIR     = 2
_COL_ENTRY   = 3
_COL_CURRENT = 4
_COL_PNL_D   = 5
_COL_PNL_R   = 6
_COL_DAYS    = 7
_COL_ALERT   = 8


class LivePositionsModel(QtCore.QAbstractTableModel):
    """Table model backed by a list of Position objects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._positions: list = []
        self._alerts: dict[int, list] = {}   # row_index → list[Alert]

    def load(self, positions: list, alerts: dict[int, list]) -> None:
        self.beginResetModel()
        self._positions = positions
        self._alerts = alerts
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self._positions)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(_HEADERS)

    def headerData(self, section: int, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation == QtCore.Qt.Orientation.Horizontal and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return _HEADERS[section]
        return None

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        pos = self._positions[index.row()]
        col = index.column()

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self._display(pos, col, index.row())

        if role == QtCore.Qt.ItemDataRole.ForegroundRole:
            return self._foreground(pos, col, index.row())

        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            if col in (_COL_ENTRY, _COL_CURRENT, _COL_PNL_D, _COL_PNL_R, _COL_DAYS):
                return QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            return QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter

        return None

    def _display(self, pos, col: int, row: int) -> str:
        if col == _COL_SYMBOL:  return pos.symbol
        if col == _COL_TYPE:    return pos.position_type.capitalize()
        if col == _COL_DIR:     return pos.direction.capitalize()
        if col == _COL_ENTRY:   return f"{pos.entry_price:.2f}"
        if col == _COL_CURRENT: return f"{pos.current_price:.2f}"
        if col == _COL_PNL_D:   return f"{pos.pnl_dollars:+.0f}"
        if col == _COL_PNL_R:
            if pos.initial_risk > 0:
                r = pos.pnl_dollars / pos.initial_risk
                return f"{r:+.2f}R"
            return "—"
        if col == _COL_DAYS:    return str(pos.days_held) if pos.days_held > 0 else "?"
        if col == _COL_ALERT:
            alerts = self._alerts.get(row, [])
            if not alerts:
                return ""
            worst = alerts[0].severity  # already sorted critical-first
            badges = {"critical": "●", "warn": "●", "ok": ""}
            return badges.get(worst, "")
        return ""

    def _foreground(self, pos, col: int, row: int):
        if col == _COL_PNL_D or col == _COL_PNL_R:
            colour = GREEN if pos.pnl_dollars >= 0 else RED
            return QtGui.QColor(colour)
        if col == _COL_ALERT:
            alerts = self._alerts.get(row, [])
            if alerts:
                colour = _SEVERITY_COLOURS.get(alerts[0].severity, DIM)
                return QtGui.QColor(colour)
        return None

    def alerts_at(self, row: int) -> list:
        return self._alerts.get(row, [])


# ---------------------------------------------------------------------------
# Alerts panel
# ---------------------------------------------------------------------------

class AlertsPanel(QtWidgets.QWidget):
    """Scrollable list of rule alerts for the selected position."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        lbl = QtWidgets.QLabel("Rule Alerts")
        lbl.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        layout.addWidget(lbl)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(self._scroll)

        self._content = QtWidgets.QWidget()
        self._inner = QtWidgets.QVBoxLayout(self._content)
        self._inner.setContentsMargins(0, 0, 0, 0)
        self._inner.setSpacing(4)
        self._inner.addStretch()
        self._scroll.setWidget(self._content)

    def load_alerts(self, alerts: list) -> None:
        # Remove old widgets
        while self._inner.count() > 1:
            item = self._inner.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not alerts:
            lbl = QtWidgets.QLabel("No alerts")
            lbl.setStyleSheet(f"color: {DIM}; font-size: 11px;")
            self._inner.insertWidget(0, lbl)
            return

        for alert in alerts:
            colour = _SEVERITY_COLOURS.get(alert.severity, DIM)
            badge = "●"
            text = f'<span style="color:{colour};">{badge}</span> ' \
                   f'<b style="color:#ccc;">[{alert.severity.upper()}]</b> {alert.message}'
            lbl = QtWidgets.QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 12px; padding: 2px 0;")
            self._inner.insertWidget(self._inner.count() - 1, lbl)


# ---------------------------------------------------------------------------
# Background refresh thread
# ---------------------------------------------------------------------------

class PositionRefreshThread(QtCore.QThread):
    """Calls fetch_live_positions() in background and emits positions_ready."""

    positions_ready = QtCore.Signal(list)
    error = QtCore.Signal(str)

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self._host = host
        self._port = port

    def run(self) -> None:
        try:
            from finance.apps.assistant._ibkr_positions import fetch_live_positions
            positions = fetch_live_positions(self._host, self._port)
            self.positions_ready.emit(positions)
        except Exception as exc:
            log.exception("PositionRefreshThread failed")
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class LivePositionsPanel(QtWidgets.QWidget):
    """
    Live Positions tab.

    Params
    ------
    ibkr_host / ibkr_port:
        TWS connection details.
    regime_status:
        Current regime string passed into evaluate_position().
    """

    def __init__(
        self,
        *,
        ibkr_host: str = "127.0.0.1",
        ibkr_port: int = 7497,
        regime_status: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._ibkr_host = ibkr_host
        self._ibkr_port = ibkr_port
        self._regime_status = regime_status
        self._refresh_thread: PositionRefreshThread | None = None

        self._build_ui()
        self._start_auto_refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Splitter: table (left) + alerts (right)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Table
        self._model = LivePositionsModel(self)
        self._table = QtWidgets.QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setShowGrid(False)
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        # Column widths
        header = self._table.horizontalHeader()
        header.resizeSection(0, 70)   # Symbol
        header.resizeSection(1, 60)   # Type
        header.resizeSection(2, 40)   # Dir
        header.resizeSection(3, 65)   # Entry
        header.resizeSection(4, 65)   # Current
        header.resizeSection(5, 65)   # P&L $
        header.resizeSection(6, 60)   # P&L R
        header.resizeSection(7, 40)   # Days
        # ⚠ stretches

        splitter.addWidget(self._table)

        # Alerts panel
        self._alerts_panel = AlertsPanel(self)
        splitter.addWidget(self._alerts_panel)
        splitter.setSizes([500, 280])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        root.addWidget(splitter, 1)

        # Bottom bar
        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)

        self._btn_refresh = QtWidgets.QPushButton("↻  Refresh")
        self._btn_refresh.setFixedWidth(100)
        self._btn_refresh.clicked.connect(self._on_refresh)
        bottom.addWidget(self._btn_refresh)

        self._lbl_updated = QtWidgets.QLabel("")
        self._lbl_updated.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        bottom.addWidget(self._lbl_updated)
        bottom.addStretch()

        root.addLayout(bottom)

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------

    def _start_auto_refresh(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_refresh)
        self._timer.start(_REFRESH_INTERVAL_MS)

    def _on_refresh(self) -> None:
        if self._refresh_thread is not None and self._refresh_thread.isRunning():
            return
        self._btn_refresh.setEnabled(False)
        self._btn_refresh.setText("⏳  Loading…")
        thread = PositionRefreshThread(self._ibkr_host, self._ibkr_port, self)
        thread.positions_ready.connect(self._on_positions_ready)
        thread.error.connect(self._on_refresh_error)
        self._refresh_thread = thread
        thread.start()

    def _on_positions_ready(self, positions: list) -> None:
        from finance.apps.assistant._rules import evaluate_position

        alerts_map: dict[int, list] = {}
        for i, pos in enumerate(positions):
            alerts_map[i] = evaluate_position(pos, self._regime_status)

        self._model.load(positions, alerts_map)
        ts = datetime.now().strftime("%H:%M:%S")
        self._lbl_updated.setText(f"Updated {ts}")
        self._btn_refresh.setEnabled(True)
        self._btn_refresh.setText("↻  Refresh")
        self._refresh_thread = None
        log.info("Live positions loaded: %d positions", len(positions))

    def _on_refresh_error(self, message: str) -> None:
        self._lbl_updated.setText(f"Error: {message[:60]}")
        self._btn_refresh.setEnabled(True)
        self._btn_refresh.setText("↻  Refresh")
        self._refresh_thread = None
        log.warning("Position refresh failed: %s", message)

    # ------------------------------------------------------------------
    # Row selection → alerts panel
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            self._alerts_panel.load_alerts([])
            return
        row = indexes[0].row()
        alerts = self._model.alerts_at(row)
        self._alerts_panel.load_alerts(alerts)

    # ------------------------------------------------------------------
    # Public API — regime can be updated from parent window
    # ------------------------------------------------------------------

    def set_regime_status(self, regime: str) -> None:
        self._regime_status = regime
