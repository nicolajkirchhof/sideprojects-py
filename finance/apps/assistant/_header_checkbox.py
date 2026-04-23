"""
finance.apps.assistant._header_checkbox
=========================================
Custom QHeaderView that renders a tristate checkbox in section 0.

The checkbox reflects the aggregate check state of the watchlist rows:
  - Unchecked  — no rows checked
  - PartiallyChecked — some rows checked
  - Checked    — all visible rows checked

Clicking section 0 toggles between fully checked and unchecked and
emits the ``toggle_all`` signal so the window can propagate the change
to the model.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

_CheckState = QtCore.Qt.CheckState
_Orientation = QtCore.Qt.Orientation


class CheckableHeader(QtWidgets.QHeaderView):
    """Horizontal header view with a tristate checkbox in column 0."""

    toggle_all = QtCore.Signal(bool)  # True = check all, False = uncheck all

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(_Orientation.Horizontal, parent)
        self._check_state: _CheckState = _CheckState.Unchecked
        self.setSectionsClickable(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_check_state(self, state: QtCore.Qt.CheckState) -> None:
        """Update the visual checkbox state without emitting toggle_all."""
        if self._check_state != state:
            self._check_state = state
            self.viewport().update()

    # ------------------------------------------------------------------
    # QHeaderView overrides
    # ------------------------------------------------------------------

    def paintSection(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        logical_index: int,
    ) -> None:
        painter.save()
        super().paintSection(painter, rect, logical_index)
        painter.restore()

        if logical_index != 0:
            return

        opt = QtWidgets.QStyleOptionButton()
        opt.rect = self._checkbox_rect(rect)
        opt.state = QtWidgets.QStyle.StateFlag.State_Enabled

        if self._check_state == _CheckState.Checked:
            opt.state |= QtWidgets.QStyle.StateFlag.State_On
        elif self._check_state == _CheckState.PartiallyChecked:
            opt.state |= QtWidgets.QStyle.StateFlag.State_NoChange
        else:
            opt.state |= QtWidgets.QStyle.StateFlag.State_Off

        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_CheckBox, opt, painter
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        logical = self.logicalIndexAt(event.pos())
        if logical == 0:
            cb_rect = self._checkbox_rect(self.sectionViewportPosition(0), self.sectionSize(0))
            if cb_rect.contains(event.pos()):
                # Toggle: if fully checked → uncheck all; otherwise → check all
                new_checked = self._check_state != _CheckState.Checked
                self.toggle_all.emit(new_checked)
                return
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _checkbox_rect(self, *args: object) -> QtCore.QRect:
        """Return the bounding rect of the checkbox within the section.

        Accepts either a QRect directly or (x_offset, width) integers.
        The checkbox is centred vertically and horizontally within a 20×20
        area inside the section.
        """
        if len(args) == 1 and isinstance(args[0], QtCore.QRect):
            section_rect: QtCore.QRect = args[0]
        else:
            x_offset = int(args[0])
            width = int(args[1])
            section_rect = QtCore.QRect(x_offset, 0, width, self.height())

        cb_size = 14
        x = section_rect.x() + (section_rect.width() - cb_size) // 2
        y = section_rect.y() + (section_rect.height() - cb_size) // 2
        return QtCore.QRect(x, y, cb_size, cb_size)
