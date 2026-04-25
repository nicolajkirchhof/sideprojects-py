"""
finance.apps.assistant._trade_review_panel
============================================
Trade Review tab widget (TA-E7-S4).

Layout:
  Left  — QListWidget of closed trades with symbol + date + P&L + rule flags
  Right — Review pane: rule evaluation summary + "Get Claude Review" button
           Claude review output (narrative + verdict + lessons)
"""
from __future__ import annotations

import logging

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

GREEN  = "#4CAF50"
AMBER  = "#FFA726"
RED    = "#F44336"
DIM    = "#666"

_VERDICT_COLOURS = {"GOOD": GREEN, "ACCEPTABLE": AMBER, "POOR": RED}
_SEVERITY_COLOURS = {"ok": GREEN, "warn": AMBER, "critical": RED}


# ---------------------------------------------------------------------------
# Background thread — Claude trade review
# ---------------------------------------------------------------------------

class TradeReviewThread(QtCore.QThread):
    """Calls Claude API for a narrative trade review."""

    review_ready = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(self, trade: dict, alerts: list, model: str, regime_at_entry: str = "", parent=None):
        super().__init__(parent)
        self._trade = trade
        self._alerts = alerts
        self._model = model
        self._regime_at_entry = regime_at_entry

    def run(self) -> None:
        try:
            result = self._call_claude()
            self.review_ready.emit(result)
        except Exception as exc:
            log.exception("TradeReviewThread failed")
            self.error.emit(str(exc))

    def _call_claude(self) -> dict:
        from finance.apps.assistant._claude import review_trade

        return review_trade(
            self._trade,
            self._alerts,
            self._model,
            regime_at_entry=self._regime_at_entry,
        )


# ---------------------------------------------------------------------------
# Review pane (right side)
# ---------------------------------------------------------------------------

class TradeReviewPane(QtWidgets.QWidget):
    """Displays rule alerts + Claude review for the selected trade."""

    review_requested = QtCore.Signal()   # emitted when user clicks "Get Claude Review"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Trade header
        self._lbl_header = QtWidgets.QLabel("")
        self._lbl_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #eee;")
        root.addWidget(self._lbl_header)

        # Rule flags section
        flags_lbl = QtWidgets.QLabel("Rule Flags")
        flags_lbl.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        root.addWidget(flags_lbl)

        self._flags_widget = QtWidgets.QWidget()
        self._flags_layout = QtWidgets.QVBoxLayout(self._flags_widget)
        self._flags_layout.setContentsMargins(0, 0, 0, 0)
        self._flags_layout.setSpacing(2)
        root.addWidget(self._flags_widget)

        # Claude review section
        claude_lbl = QtWidgets.QLabel("Claude Review")
        claude_lbl.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        root.addWidget(claude_lbl)

        self._review_text = QtWidgets.QTextEdit()
        self._review_text.setReadOnly(True)
        self._review_text.setStyleSheet("font-size: 12px; background: #1a1a2e; border: 1px solid #333;")
        root.addWidget(self._review_text, 1)

        # Button row
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_review = QtWidgets.QPushButton("Get Claude Review")
        self._btn_review.setEnabled(False)
        self._btn_review.clicked.connect(self.review_requested.emit)
        btn_row.addWidget(self._btn_review)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def load_trade(self, trade: dict | None, alerts: list) -> None:
        if trade is None:
            self._lbl_header.setText("")
            self._clear_flags()
            self._review_text.clear()
            self._btn_review.setEnabled(False)
            return

        symbol = trade.get("symbol", "?")
        pnl = trade.get("pnl") or trade.get("pnlDollars") or 0
        colour = GREEN if float(pnl) >= 0 else RED
        self._lbl_header.setText(
            f'<span style="color:#eee;">{symbol}</span>  '
            f'<span style="color:{colour};">{float(pnl):+.0f}$</span>'
        )

        self._clear_flags()
        if alerts:
            for alert in alerts:
                c = _SEVERITY_COLOURS.get(alert.severity, DIM)
                lbl = QtWidgets.QLabel(
                    f'<span style="color:{c};">●</span> '
                    f'<b>[{alert.severity.upper()}]</b> {alert.message}'
                )
                lbl.setWordWrap(True)
                lbl.setStyleSheet("font-size: 12px;")
                self._flags_layout.addWidget(lbl)
        else:
            lbl = QtWidgets.QLabel("No rule flags")
            lbl.setStyleSheet(f"color: {DIM}; font-size: 12px;")
            self._flags_layout.addWidget(lbl)

        self._review_text.clear()
        self._btn_review.setEnabled(True)

    def _clear_flags(self) -> None:
        while self._flags_layout.count():
            item = self._flags_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def set_review_generating(self) -> None:
        self._review_text.setHtml(f'<span style="color:{DIM};">Generating review…</span>')
        self._btn_review.setEnabled(False)

    def load_review(self, review: dict) -> None:
        self._btn_review.setEnabled(True)
        verdict = review.get("verdict", "")
        v_colour = _VERDICT_COLOURS.get(verdict, DIM)
        summary = review.get("summary", "")
        well = review.get("what_went_well") or []
        improve = review.get("what_to_improve") or []
        lesson = review.get("key_lesson", "")

        parts = []
        if verdict:
            parts.append(f'<b style="color:{v_colour};">Verdict: {verdict}</b><br>')
        if summary:
            parts.append(f'{summary}<br>')
        if well:
            parts.append('<b>What went well:</b>')
            parts.extend(f'  ✓ {item}' for item in well)
            parts.append("")
        if improve:
            parts.append('<b>What to improve:</b>')
            parts.extend(f'  ✗ {item}' for item in improve)
            parts.append("")
        if lesson:
            parts.append(f'<b>Key lesson:</b> {lesson}')

        self._review_text.setHtml("<br>".join(parts))

    def set_review_error(self, message: str) -> None:
        self._review_text.setHtml(f'<span style="color:{RED};">Error: {message}</span>')
        self._btn_review.setEnabled(True)


# ---------------------------------------------------------------------------
# Closed trade list item helper
# ---------------------------------------------------------------------------

def _make_trade_label(trade: dict, alerts: list) -> str:
    symbol = trade.get("symbol", "?")
    pnl = float(trade.get("pnl") or trade.get("pnlDollars") or 0)
    close_date = str(trade.get("closeDate") or "")[:10]
    flag = " ⚠" if any(a.severity == "critical" for a in alerts) else ""
    return f"{symbol}  {close_date}  {pnl:+.0f}${flag}"


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class TradeReviewPanel(QtWidgets.QWidget):
    """
    Trade Review tab.

    Loads closed trades from the Tradelog API, evaluates each with the
    rules engine, and allows requesting a Claude narrative review.
    """

    def __init__(
        self,
        *,
        tradelog_base_url: str = "http://localhost:5186",
        claude_model: str = "claude-haiku-4-5-20251001",
        regime_status: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._tradelog_url = tradelog_base_url
        self._claude_model = claude_model
        self._regime_status = regime_status
        self._trades: list[dict] = []
        self._trade_alerts: list[list] = []
        self._review_thread: TradeReviewThread | None = None

        self._build_ui()
        self._load_trades()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left — trade list
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        lbl = QtWidgets.QLabel("Closed Trades")
        lbl.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        left_layout.addWidget(lbl)

        self._trade_list = QtWidgets.QListWidget()
        self._trade_list.setStyleSheet("font-family: 'Roboto Mono', monospace; font-size: 12px;")
        self._trade_list.currentRowChanged.connect(self._on_trade_selected)
        left_layout.addWidget(self._trade_list, 1)

        self._btn_reload = QtWidgets.QPushButton("↻  Reload")
        self._btn_reload.setFixedWidth(90)
        self._btn_reload.clicked.connect(self._load_trades)
        left_layout.addWidget(self._btn_reload)

        splitter.addWidget(left)

        # Right — review pane
        self._review_pane = TradeReviewPane(self)
        self._review_pane.review_requested.connect(self._on_review_requested)
        splitter.addWidget(self._review_pane)

        splitter.setSizes([300, 500])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_trades(self) -> None:
        from finance.apps.assistant._tradelog_client import fetch_closed_trades
        from finance.apps.assistant._rules import evaluate_position, Position

        self._btn_reload.setEnabled(False)
        trades = fetch_closed_trades(self._tradelog_url)

        self._trades = trades
        self._trade_alerts = []
        self._trade_list.clear()

        for trade in trades:
            pos = self._trade_to_position(trade)
            alerts: list = []
            if pos is not None:
                alerts = evaluate_position(pos, self._regime_status)
            self._trade_alerts.append(alerts)

            label = _make_trade_label(trade, alerts)
            item = QtWidgets.QListWidgetItem(label)

            # Colour critical trades in list
            if any(a.severity == "critical" for a in alerts):
                item.setForeground(QtGui.QColor(RED))
            elif any(a.severity == "warn" for a in alerts):
                item.setForeground(QtGui.QColor(AMBER))

            self._trade_list.addItem(item)

        self._btn_reload.setEnabled(True)
        log.info("Trade Review: loaded %d closed trades", len(trades))

    @staticmethod
    def _trade_to_position(trade: dict):
        """Map a Tradelog closed trade dict to a Position for rule evaluation."""
        from finance.apps.assistant._rules import Position

        symbol = trade.get("symbol", "")
        if not symbol:
            return None

        pnl = float(trade.get("pnl") or 0)
        open_price = float(trade.get("entryPrice") or trade.get("openPrice") or 0)
        close_price = float(trade.get("exitPrice") or trade.get("closePrice") or 0)

        # XAtrMove as 1R proxy for stocks
        initial_risk = float(trade.get("xAtrMove") or trade.get("initialRisk") or 1)

        days_held = int(trade.get("daysHeld") or 0)
        direction = str(trade.get("direction") or "long").lower()
        if direction not in ("long", "short"):
            direction = "long"

        position_type = str(trade.get("positionType") or trade.get("type") or "stock").lower()
        if position_type not in ("stock", "option"):
            position_type = "stock"

        return Position(
            symbol=symbol,
            position_type=position_type,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            entry_price=open_price,
            current_price=close_price,
            initial_risk=max(initial_risk, 0.01),
            pnl_dollars=pnl,
            days_held=days_held,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_trade_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._trades):
            self._review_pane.load_trade(None, [])
            return
        trade = self._trades[row]
        alerts = self._trade_alerts[row] if row < len(self._trade_alerts) else []
        self._review_pane.load_trade(trade, alerts)

    def _on_review_requested(self) -> None:
        if self._review_thread is not None and self._review_thread.isRunning():
            return
        row = self._trade_list.currentRow()
        if row < 0 or row >= len(self._trades):
            return

        trade = self._trades[row]
        alerts = self._trade_alerts[row] if row < len(self._trade_alerts) else []

        self._review_pane.set_review_generating()
        thread = TradeReviewThread(
            trade=trade,
            alerts=alerts,
            model=self._claude_model,
            regime_at_entry=self._regime_status,
            parent=self,
        )
        thread.review_ready.connect(self._on_review_ready)
        thread.error.connect(self._on_review_error)
        self._review_thread = thread
        thread.start()

    def _on_review_ready(self, review: dict) -> None:
        self._review_pane.load_review(review)
        self._review_thread = None

    def _on_review_error(self, message: str) -> None:
        self._review_pane.set_review_error(message)
        self._review_thread = None
        log.warning("Trade review failed: %s", message)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_regime_status(self, regime: str) -> None:
        self._regime_status = regime
