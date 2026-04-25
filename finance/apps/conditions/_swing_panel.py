"""Left panel — Swing Trading Regime (Epic 1: E1-S1, E1-S2)."""
from __future__ import annotations

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

from finance.apps.conditions._data import TrendStatus, VixStatus

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
# SwingRegimePanel — assembles everything
# ---------------------------------------------------------------------------

class SwingRegimePanel(QtWidgets.QWidget):
    """Left panel: SPY/QQQ trend indicators, VIX, and composite GO/NO-GO."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
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

        layout.addStretch()

    def update_indicators(
        self,
        spy: TrendStatus | None,
        qqq: TrendStatus | None,
        vix: VixStatus | None,
        status: str,
    ) -> None:
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

        self.go_nogo.set_status(status)
