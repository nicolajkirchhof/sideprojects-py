"""
finance.apps.assistant._swing_panel
======================================
Left panel — Swing Trading Regime indicators.

Shows: SPY/QQQ trend rows, VIX row, composite GO/NO-GO banner,
stop-out counter, macro event proximity warning, and economic
events calendar.

Migrated from finance.apps.conditions._swing_panel; extended for
TA-E3-S2 (stop-out counter) and TA-E3-S3 (events calendar).
"""
from __future__ import annotations

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from finance.apps.assistant._data import TrendStatus, VixStatus

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

GREEN = "#4CAF50"
AMBER = "#FFA726"
RED = "#F44336"
DIM = "#666"

_ZONE_COLOURS = {"low": GREEN, "elevated": AMBER, "high": RED}
_SLOPE_ARROWS = {"rising": "\u2197", "flat": "\u2192", "falling": "\u2198"}  # ↗ → ↘
_DIRECTION_ARROWS = {"falling": "\u2193", "rising": "\u2191", "spiking": "\u26a0"}  # ↓ ↑ ⚠

_GONOGO_STYLES = {
    "GO": (GREEN, "#111"),
    "CAUTION": (AMBER, "#111"),
    "NO-GO": (RED, "#fff"),
}

MONO_FONT = "Roboto Mono"


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------

def _dot(colour: str) -> str:
    return f'<span style="color:{colour}; font-size:18px;">\u25cf</span>'


def _label(text: str = "", mono: bool = False, size: int = 13) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    font_family = MONO_FONT if mono else "Segoe UI"
    lbl.setFont(QtGui.QFont(font_family, size))
    lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
    return lbl


def _hsep() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setStyleSheet("color: #333;")
    return line


# ---------------------------------------------------------------------------
# TrendRow — one row per symbol (SPY / QQQ)
# ---------------------------------------------------------------------------

class TrendRow(QtWidgets.QWidget):
    """Displays: SYMBOL  price  50d SMA ●↗  200d SMA ●↗"""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(12)

        self._sym = _label(mono=False, size=13)
        self._sym.setFixedWidth(48)
        self._sym.setStyleSheet("font-weight: bold;")
        h.addWidget(self._sym)

        self._price = _label(mono=True, size=13)
        self._price.setFixedWidth(80)
        h.addWidget(self._price)

        self._sma50 = _label(mono=True, size=12)
        self._sma50.setFixedWidth(170)
        h.addWidget(self._sma50)

        self._sma200 = _label(mono=True, size=12)
        self._sma200.setFixedWidth(170)
        h.addWidget(self._sma200)

        h.addStretch()

    def update_data(self, ts: TrendStatus) -> None:
        self._sym.setText(ts.symbol)

        self._price.setText(f"{ts.last_price:,.2f}")

        # 50d SMA
        col_50 = GREEN if ts.price_above_50 else RED
        arrow_50 = _SLOPE_ARROWS.get(ts.sma_50_slope, "")
        self._sma50.setText(
            f"50d {ts.sma_50:,.1f} {_dot(col_50)} "
            f'<span style="color:{col_50};">{arrow_50}</span>'
        )

        # 200d SMA
        col_200 = GREEN if ts.price_above_200 else RED
        arrow_200 = _SLOPE_ARROWS.get(ts.sma_200_slope, "")
        self._sma200.setText(
            f"200d {ts.sma_200:,.1f} {_dot(col_200)} "
            f'<span style="color:{col_200};">{arrow_200}</span>'
        )

    def clear_data(self) -> None:
        for lbl in (self._price, self._sma50, self._sma200):
            lbl.setText('<span style="color:#555;">—</span>')


# ---------------------------------------------------------------------------
# VixRow
# ---------------------------------------------------------------------------

class VixRow(QtWidgets.QWidget):
    """Displays: VIX  level  zone-badge  direction-arrow"""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(12)

        sym = _label("VIX", mono=False, size=13)
        sym.setFixedWidth(48)
        sym.setStyleSheet("font-weight: bold;")
        h.addWidget(sym)

        self._level = _label(mono=True, size=13)
        self._level.setFixedWidth(80)
        h.addWidget(self._level)

        self._badge = _label(size=12)
        self._badge.setFixedWidth(100)
        self._badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        h.addWidget(self._badge)

        self._direction = _label(size=14)
        self._direction.setFixedWidth(40)
        h.addWidget(self._direction)

        h.addStretch()

    def update_data(self, vs: VixStatus) -> None:
        colour = _ZONE_COLOURS.get(vs.zone, DIM)

        self._level.setText(f"{vs.level:.2f}")
        self._level.setStyleSheet(f"color: {colour};")

        zone_text = vs.zone.upper()
        self._badge.setText(zone_text)
        self._badge.setStyleSheet(
            f"background: {colour}; color: #111; padding: 2px 8px; "
            f"border-radius: 3px; font-weight: bold;"
        )

        arrow = _DIRECTION_ARROWS.get(vs.direction, "")
        self._direction.setText(arrow)
        self._direction.setStyleSheet(f"color: {colour}; font-size: 16px;")

    def clear_data(self) -> None:
        self._level.setText('<span style="color:#555;">—</span>')
        self._badge.setText("")
        self._badge.setStyleSheet("")
        self._direction.setText("")


# ---------------------------------------------------------------------------
# GoNoGoBar
# ---------------------------------------------------------------------------

class GoNoGoBar(QtWidgets.QLabel):
    """Large coloured badge showing GO / CAUTION / NO-GO."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(56)
        self.setFont(QtGui.QFont("Segoe UI", 20, QtGui.QFont.Weight.Bold))
        self.set_status("NO-GO")

    def set_status(self, status: str) -> None:
        bg, fg = _GONOGO_STYLES.get(status, (DIM, "#fff"))
        self.setText(status)
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 6px; "
            f"padding: 4px 16px;"
        )


# ---------------------------------------------------------------------------
# StopOutCounter
# ---------------------------------------------------------------------------

_STOPOUT_NOGO_THRESHOLD: int = 3


class StopOutCounter(QtWidgets.QWidget):
    """Manual consecutive stop-out counter.

    Displays a -/count/+ row.  When count reaches _STOPOUT_NOGO_THRESHOLD
    the override is active and the GO/NO-GO banner is forced to NO-GO.
    In-session only — resets on app restart.
    """

    count_changed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._count: int = 0

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(8)

        lbl = _label("Stop-outs:", size=12)
        h.addWidget(lbl)

        self._btn_dec = QtWidgets.QToolButton()
        self._btn_dec.setText("−")
        self._btn_dec.setFixedSize(22, 22)
        self._btn_dec.clicked.connect(self.decrement)
        h.addWidget(self._btn_dec)

        self._count_lbl = _label("0", mono=True, size=13)
        self._count_lbl.setFixedWidth(28)
        self._count_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        h.addWidget(self._count_lbl)

        self._btn_inc = QtWidgets.QToolButton()
        self._btn_inc.setText("+")
        self._btn_inc.setFixedSize(22, 22)
        self._btn_inc.clicked.connect(self.increment)
        h.addWidget(self._btn_inc)

        h.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_override_active(self) -> bool:
        return self._count >= _STOPOUT_NOGO_THRESHOLD

    def increment(self) -> None:
        self._count += 1
        self._refresh()

    def decrement(self) -> None:
        if self._count > 0:
            self._count -= 1
            self._refresh()

    def reset(self) -> None:
        self._count = 0
        self._refresh()

    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        colour = RED if self.is_override_active else "#ccc"
        self._count_lbl.setText(str(self._count))
        self._count_lbl.setStyleSheet(f"color: {colour}; font-weight: bold;")
        self.count_changed.emit(self._count)


# ---------------------------------------------------------------------------
# MacroEventLabel — inline risk text above GO/NO-GO bar
# ---------------------------------------------------------------------------

class MacroEventLabel(QtWidgets.QLabel):
    """Single-line label showing the nearest high-impact macro event risk."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setFont(QtGui.QFont("Segoe UI", 11))
        self.setStyleSheet("color: #aaa;")
        self.hide()

    def set_events(self, event_dicts: list[dict]) -> None:
        """Update from a list of serialised event dicts (from cache or pipeline)."""
        from finance.apps.assistant._calendar import check_macro_risk_from_dicts
        has_risk, risks = check_macro_risk_from_dicts(event_dicts)
        if has_risk and risks:
            # Show only the most urgent risk
            text = risks[0]
            if text.startswith("NO-GO"):
                self.setStyleSheet(f"color: {RED};")
            elif text.startswith("CAUTION"):
                self.setStyleSheet(f"color: {AMBER};")
            else:
                self.setStyleSheet("color: #aaa;")
            self.setText(text)
            self.show()
        else:
            self.hide()


# ---------------------------------------------------------------------------
# EventsCalendarWidget — compact events table
# ---------------------------------------------------------------------------

_MAX_EVENTS_SHOWN: int = 10


class EventsCalendarWidget(QtWidgets.QWidget):
    """Compact scrollable list of upcoming high-impact economic events.

    Each entry shows: [impact colour] title / date-time.
    Events within 48 hours are highlighted.
    NO-GO keywords get a red warning colour.
    Max _MAX_EVENTS_SHOWN rows.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = _label("ECONOMIC EVENTS (5d, High)", size=11)
        header.setStyleSheet("color: #888;")
        layout.addWidget(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(160)

        self._container = QtWidgets.QWidget()
        self._container_layout = QtWidgets.QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(3)
        self._container_layout.addStretch()

        scroll.setWidget(self._container)
        layout.addWidget(scroll)

        self._placeholder = _label("No events loaded", size=11)
        self._placeholder.setStyleSheet("color: #555;")
        self._container_layout.insertWidget(0, self._placeholder)

    def set_events(self, event_dicts: list[dict]) -> None:
        """Rebuild the event list from serialised event dicts."""
        from datetime import datetime, timezone

        # Clear existing rows (keep stretch at end)
        while self._container_layout.count() > 1:
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        now = datetime.now(timezone.utc)
        shown = 0

        for d in event_dicts[:_MAX_EVENTS_SHOWN]:
            try:
                dt = datetime.fromisoformat(d["date"])
            except (ValueError, KeyError):
                continue

            hours_away = (dt - now).total_seconds() / 3600
            title = d.get("title", "")
            country = d.get("country", "")

            from finance.apps.assistant._calendar import NO_GO_KEYWORDS
            is_nogo = any(kw.lower() in title.lower() for kw in NO_GO_KEYWORDS)
            is_imminent = hours_away <= 48

            if is_nogo and is_imminent:
                colour = RED
            elif is_imminent:
                colour = AMBER
            else:
                colour = "#888"

            date_str = dt.strftime("%a %b %d %H:%M")
            row_lbl = _label(f"{title}\n{country}  {date_str}", size=11)
            row_lbl.setWordWrap(True)
            row_lbl.setStyleSheet(f"color: {colour};")
            self._container_layout.insertWidget(shown, row_lbl)
            shown += 1

        if shown == 0:
            placeholder = _label("No upcoming high-impact events", size=11)
            placeholder.setStyleSheet("color: #555;")
            self._container_layout.insertWidget(0, placeholder)


# ---------------------------------------------------------------------------
# SwingRegimePanel — assembles everything
# ---------------------------------------------------------------------------

class SwingRegimePanel(QtWidgets.QWidget):
    """Left panel: SPY/QQQ trend indicators, VIX, GO/NO-GO with stop-out
    override, macro event warning, and events calendar.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._natural_status: str = "NO-GO"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        header = QtWidgets.QLabel("SWING REGIME")
        header.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Weight.Bold))
        header.setStyleSheet("color: #ccc;")
        layout.addWidget(header)

        self.spy_row = TrendRow()
        self.qqq_row = TrendRow()
        layout.addWidget(self.spy_row)
        layout.addWidget(self.qqq_row)

        layout.addWidget(_hsep())

        self.vix_row = VixRow()
        layout.addWidget(self.vix_row)

        layout.addWidget(_hsep())

        self.go_nogo = GoNoGoBar()
        layout.addWidget(self.go_nogo)

        layout.addWidget(_hsep())

        self.stop_counter = StopOutCounter()
        self.stop_counter.count_changed.connect(self._on_stopout_changed)
        layout.addWidget(self.stop_counter)

        self.macro_label = MacroEventLabel()
        layout.addWidget(self.macro_label)

        layout.addWidget(_hsep())

        self.events_widget = EventsCalendarWidget()
        layout.addWidget(self.events_widget)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_indicators(
        self,
        spy: TrendStatus | None,
        qqq: TrendStatus | None,
        vix: VixStatus | None,
        status: str,
    ) -> None:
        self._natural_status = status

        if spy:
            self.spy_row.update_data(spy)
        else:
            self.spy_row.clear_data()

        if qqq:
            self.qqq_row.update_data(qqq)
        else:
            self.qqq_row.clear_data()

        if vix:
            self.vix_row.update_data(vix)
        else:
            self.vix_row.clear_data()

        self._apply_status()

    def update_events(self, event_dicts: list[dict]) -> None:
        """Update macro warning label and events calendar from cache/pipeline data."""
        self.macro_label.set_events(event_dicts)
        self.events_widget.set_events(event_dicts)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_stopout_changed(self, _count: int) -> None:
        self._apply_status()

    def _apply_status(self) -> None:
        """Apply GO/NO-GO, respecting stop-out override."""
        if self.stop_counter.is_override_active:
            self.go_nogo.set_status("NO-GO")
        else:
            self.go_nogo.set_status(self._natural_status)
