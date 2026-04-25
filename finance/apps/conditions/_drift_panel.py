"""Placeholder right panel for the DRIFT regime (Epic 2)."""
from __future__ import annotations

from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


class DriftPanel(QtWidgets.QWidget):
    """Placeholder — will hold DRIFT regime tier, underlying registry, etc."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QtWidgets.QLabel("DRIFT REGIME")
        header.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Weight.Bold))
        header.setStyleSheet("color: #888;")
        layout.addWidget(header)

        placeholder = QtWidgets.QLabel("Coming in Epic 2")
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #555; font-size: 13px;")
        layout.addWidget(placeholder, 1)
