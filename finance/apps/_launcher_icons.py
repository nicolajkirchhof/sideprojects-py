"""
finance.apps._launcher_icons
==============================
Programmatic icon generation for the app launcher.

Each icon is drawn via QPainter onto a QPixmap — no external image files needed.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtGui, QtCore

_ICON_SIZE = 64


def _make_pixmap() -> tuple[QtGui.QPixmap, QtGui.QPainter]:
    pm = QtGui.QPixmap(_ICON_SIZE, _ICON_SIZE)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    return pm, p


def _candlestick_icon() -> QtGui.QIcon:
    """Swing-plot: three candlesticks (green up, red down, green up)."""
    pm, p = _make_pixmap()

    candles = [
        # (x, open_y, close_y, high_y, low_y, color)
        (12, 38, 18, 10, 46, QtGui.QColor("#4CAF50")),
        (28, 20, 40, 12, 50, QtGui.QColor("#F44336")),
        (44, 42, 16, 8,  52, QtGui.QColor("#4CAF50")),
    ]
    bar_w = 10
    for x, oy, cy, hy, ly, color in candles:
        # Wick
        pen = QtGui.QPen(color, 2)
        p.setPen(pen)
        cx = x + bar_w // 2
        p.drawLine(cx, hy, cx, ly)
        # Body
        p.setBrush(color)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        top = min(oy, cy)
        h = abs(oy - cy) or 2
        p.drawRect(x, top, bar_w, h)

    p.end()
    return QtGui.QIcon(pm)


def _momentum_icon() -> QtGui.QIcon:
    """Momentum dashboard: rising bar chart with arrow."""
    pm, p = _make_pixmap()

    bars = [
        (8, 44, 12, QtGui.QColor("#5C6BC0")),
        (22, 34, 12, QtGui.QColor("#42A5F5")),
        (36, 20, 12, QtGui.QColor("#26C6DA")),
        (50, 10, 10, QtGui.QColor("#66BB6A")),
    ]
    for x, y, w, color in bars:
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(x, y, w, 56 - y, 2, 2)

    # Arrow pointing up-right
    pen = QtGui.QPen(QtGui.QColor("#EEEEEE"), 2.5)
    p.setPen(pen)
    p.drawLine(14, 40, 54, 10)
    # Arrowhead
    p.setBrush(QtGui.QColor("#EEEEEE"))
    arrow = QtGui.QPolygonF([
        QtCore.QPointF(54, 10),
        QtCore.QPointF(44, 12),
        QtCore.QPointF(52, 20),
    ])
    p.drawPolygon(arrow)

    p.end()
    return QtGui.QIcon(pm)


def _generic_icon(letter: str) -> QtGui.QIcon:
    """Fallback: rounded square with first letter."""
    pm, p = _make_pixmap()

    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QColor("#37474F"))
    p.drawRoundedRect(4, 4, 56, 56, 10, 10)

    p.setPen(QtGui.QColor("#EEEEEE"))
    font = QtGui.QFont("Segoe UI", 28, QtGui.QFont.Weight.Bold)
    p.setFont(font)
    p.drawText(QtCore.QRect(4, 4, 56, 56), QtCore.Qt.AlignmentFlag.AlignCenter, letter.upper())

    p.end()
    return QtGui.QIcon(pm)


# Map APP_ICON_ID values to painter functions
_ICON_REGISTRY: dict[str, callable] = {
    "candlestick": _candlestick_icon,
    "momentum": _momentum_icon,
}


def icon_for_app(icon_id: str | None, app_name: str) -> QtGui.QIcon:
    """Return an icon for the given app, falling back to a generic letter icon."""
    if icon_id and icon_id in _ICON_REGISTRY:
        return _ICON_REGISTRY[icon_id]()
    return _generic_icon(app_name[0])
