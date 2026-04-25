"""
finance.apps.assistant._filter_bar
=====================================
Filter bar widget for the watchlist table — TA-E4-S2.

FilterBar sits above the watchlist QTableView. It emits filters_changed
whenever any control changes so the caller can push a new FilterState to
the proxy model.

Layout
------
  [Score ≥ N] [≤ N] [Dir ▼] [Tags (n) ▼] [Sectors (n) ▼] [Search ___] [Reset]
  Active: score 40–70  |  dir: long  |  ...   (hidden when no filters active)

Tag and sector multi-select use a QToolButton with a detached QMenu whose
actions are checkable.  The menu stays open after each click via a
_MultiSelectMenu subclass that overrides mouseReleaseEvent.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # QAction is in QtGui for PyQt6

from finance.apps.assistant._filter_proxy import (
    FilterState,
    SCORE_MAX_DEFAULT,
    SCORE_MIN_DEFAULT,
)

_BUTTON_STYLE = """
QToolButton {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 2px 6px;
    color: #ccc;
    font-size: 11px;
}
QToolButton:hover { background: #333; }
QToolButton:checked { background: #1a3a1a; border-color: #4CAF50; color: #4CAF50; }
"""

_SPINBOX_STYLE = """
QSpinBox {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 2px 4px;
    color: #ccc;
    font-size: 11px;
}
QSpinBox::up-button, QSpinBox::down-button { width: 14px; }
"""

_SEARCH_STYLE = """
QLineEdit {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 2px 6px;
    color: #ccc;
    font-size: 11px;
}
QLineEdit:focus { border-color: #4CAF50; }
"""

_LABEL_STYLE = "color: #777; font-size: 11px;"
_ACTIVE_STYLE = "color: #FFA726; font-size: 11px; padding: 2px 0;"


# ---------------------------------------------------------------------------
# _MultiSelectMenu — QMenu that stays open after item click
# ---------------------------------------------------------------------------


class _MultiSelectMenu(QtWidgets.QMenu):
    """
    QMenu subclass that keeps itself open after each action is toggled
    so the user can select multiple items without reopening the menu.
    """

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        action = self.activeAction()
        if action and action.isCheckable():
            action.trigger()
            # Do not call super() — prevents the menu from closing.
        else:
            super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# FilterBar
# ---------------------------------------------------------------------------


class FilterBar(QtWidgets.QWidget):
    """
    Horizontal filter bar for the watchlist table.

    Signals
    -------
    filters_changed(FilterState)
        Emitted whenever any filter control changes.  The caller should
        push the new state to WatchlistFilterProxy.update_filter().
    """

    filters_changed = QtCore.Signal(object)  # FilterState

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tag_actions: dict[str, QtGui.QAction] = {}
        self._sector_actions: dict[str, QtGui.QAction] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_options(
        self,
        tags_with_counts: dict[str, int],
        sectors_with_counts: dict[str, int],
    ) -> None:
        """
        Rebuild tag and sector menus from freshly loaded data.

        *tags_with_counts* maps tag name → count of rows carrying that tag.
        *sectors_with_counts* maps sector name → count of rows in that sector.
        Existing selections are cleared when options change.
        """
        self._rebuild_menu(
            self._tag_menu,
            self._tag_actions,
            tags_with_counts,
            self._btn_tags,
            "Tags",
        )
        self._rebuild_menu(
            self._sector_menu,
            self._sector_actions,
            sectors_with_counts,
            self._btn_sectors,
            "Sectors",
        )
        # Selections were cleared by _rebuild_menu; emit so the proxy
        # reflects the now-empty tag/sector sets immediately.
        self._emit()

    def current_state(self) -> FilterState:
        """Return a FilterState reflecting the current control values."""
        return FilterState(
            score_min=float(self._spin_min.value()),
            score_max=float(self._spin_max.value()),
            direction=self._current_direction(),
            tags=frozenset(
                name for name, act in self._tag_actions.items() if act.isChecked()
            ),
            sectors=frozenset(
                name for name, act in self._sector_actions.items() if act.isChecked()
            ),
            text=self._search.text().strip(),
        )

    def reset(self) -> None:
        """Reset all controls to defaults without emitting an intermediate signal."""
        self._spin_min.blockSignals(True)
        self._spin_max.blockSignals(True)
        self._spin_min.setValue(int(SCORE_MIN_DEFAULT))
        self._spin_max.setValue(int(SCORE_MAX_DEFAULT))
        self._spin_min.blockSignals(False)
        self._spin_max.blockSignals(False)

        for btn in (self._btn_long, self._btn_short):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

        for act in self._tag_actions.values():
            act.setChecked(False)
        for act in self._sector_actions.values():
            act.setChecked(False)

        self._search.blockSignals(True)
        self._search.clear()
        self._search.blockSignals(False)

        self._update_button_labels()
        self._emit()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 2)
        outer.setSpacing(2)

        # --- Controls row ---
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)

        # Score range
        score_lbl = QtWidgets.QLabel("Score")
        score_lbl.setStyleSheet(_LABEL_STYLE)
        self._spin_min = QtWidgets.QSpinBox()
        self._spin_min.setRange(0, int(SCORE_MAX_DEFAULT))
        self._spin_min.setValue(int(SCORE_MIN_DEFAULT))
        self._spin_min.setFixedWidth(50)
        self._spin_min.setStyleSheet(_SPINBOX_STYLE)
        self._spin_min.setToolTip("Minimum score (inclusive)")

        sep_lbl = QtWidgets.QLabel("–")
        sep_lbl.setStyleSheet(_LABEL_STYLE)

        self._spin_max = QtWidgets.QSpinBox()
        self._spin_max.setRange(0, int(SCORE_MAX_DEFAULT))
        self._spin_max.setValue(int(SCORE_MAX_DEFAULT))
        self._spin_max.setFixedWidth(50)
        self._spin_max.setStyleSheet(_SPINBOX_STYLE)
        self._spin_max.setToolTip("Maximum score (inclusive)")

        controls.addWidget(score_lbl)
        controls.addWidget(self._spin_min)
        controls.addWidget(sep_lbl)
        controls.addWidget(self._spin_max)

        # Direction toggle buttons
        self._btn_long = QtWidgets.QToolButton()
        self._btn_long.setText("Long")
        self._btn_long.setCheckable(True)
        self._btn_long.setStyleSheet(_BUTTON_STYLE)
        self._btn_long.setToolTip("Show long candidates only")

        self._btn_short = QtWidgets.QToolButton()
        self._btn_short.setText("Short")
        self._btn_short.setCheckable(True)
        self._btn_short.setStyleSheet(_BUTTON_STYLE)
        self._btn_short.setToolTip("Show short candidates only")

        controls.addWidget(self._btn_long)
        controls.addWidget(self._btn_short)

        # Tags multi-select
        self._tag_menu = _MultiSelectMenu(self)
        self._btn_tags = QtWidgets.QToolButton()
        self._btn_tags.setText("Tags")
        self._btn_tags.setStyleSheet(_BUTTON_STYLE)
        self._btn_tags.setToolTip("Filter by tag (OR)")
        self._btn_tags.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_tags.setMenu(self._tag_menu)
        controls.addWidget(self._btn_tags)

        # Sectors multi-select
        self._sector_menu = _MultiSelectMenu(self)
        self._btn_sectors = QtWidgets.QToolButton()
        self._btn_sectors.setText("Sectors")
        self._btn_sectors.setStyleSheet(_BUTTON_STYLE)
        self._btn_sectors.setToolTip("Filter by sector (OR)")
        self._btn_sectors.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_sectors.setMenu(self._sector_menu)
        controls.addWidget(self._btn_sectors)

        # Text search
        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("Search symbol / sector…")
        self._search.setFixedWidth(180)
        self._search.setStyleSheet(_SEARCH_STYLE)
        controls.addWidget(self._search)

        controls.addStretch()

        # Reset button
        self._btn_reset = QtWidgets.QToolButton()
        self._btn_reset.setText("Reset")
        self._btn_reset.setStyleSheet(_BUTTON_STYLE)
        self._btn_reset.setToolTip("Clear all filters")
        controls.addWidget(self._btn_reset)

        outer.addLayout(controls)

        # --- Active filters summary ---
        self._lbl_active = QtWidgets.QLabel("")
        self._lbl_active.setStyleSheet(_ACTIVE_STYLE)
        self._lbl_active.setVisible(False)
        outer.addWidget(self._lbl_active)

        # Connections
        self._spin_min.valueChanged.connect(self._on_score_changed)
        self._spin_max.valueChanged.connect(self._on_score_changed)
        self._btn_long.toggled.connect(self._on_direction_toggled)
        self._btn_short.toggled.connect(self._on_direction_toggled)
        self._tag_menu.triggered.connect(self._on_menu_triggered)
        self._sector_menu.triggered.connect(self._on_menu_triggered)
        self._search.textChanged.connect(self._on_text_changed)
        self._btn_reset.clicked.connect(self.reset)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_direction(self) -> str:
        if self._btn_long.isChecked() and not self._btn_short.isChecked():
            return "long"
        if self._btn_short.isChecked() and not self._btn_long.isChecked():
            return "short"
        return "all"

    def _rebuild_menu(
        self,
        menu: QtWidgets.QMenu,
        actions: dict[str, QtGui.QAction],
        items_with_counts: dict[str, int],
        button: QtWidgets.QToolButton,
        label: str,
    ) -> None:
        """Replace all actions in *menu* with new entries from *items_with_counts*."""
        menu.clear()
        actions.clear()
        for name in sorted(items_with_counts):
            count = items_with_counts[name]
            action = QtGui.QAction(f"{name}  ({count})", self)
            action.setCheckable(True)
            menu.addAction(action)
            actions[name] = action
        button.setText(label if not items_with_counts else f"{label}  ▾")

    def _update_button_labels(self) -> None:
        """Refresh tag/sector button text to show selected count."""
        checked_tags = sum(1 for a in self._tag_actions.values() if a.isChecked())
        if checked_tags:
            self._btn_tags.setText(f"Tags ({checked_tags}) ▾")
        elif self._tag_actions:
            self._btn_tags.setText("Tags  ▾")
        else:
            self._btn_tags.setText("Tags")

        checked_sectors = sum(1 for a in self._sector_actions.values() if a.isChecked())
        if checked_sectors:
            self._btn_sectors.setText(f"Sectors ({checked_sectors}) ▾")
        elif self._sector_actions:
            self._btn_sectors.setText("Sectors  ▾")
        else:
            self._btn_sectors.setText("Sectors")

    def _emit(self) -> None:
        """Build current FilterState, update active label, emit signal."""
        state = self.current_state()
        summary = state.active_summary()
        if summary:
            self._lbl_active.setText(summary)
            self._lbl_active.setVisible(True)
        else:
            self._lbl_active.setVisible(False)
        self.filters_changed.emit(state)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_score_changed(self) -> None:
        # Keep min ≤ max
        if self._spin_min.value() > self._spin_max.value():
            sender = self.sender()
            if sender is self._spin_min:
                self._spin_max.blockSignals(True)
                self._spin_max.setValue(self._spin_min.value())
                self._spin_max.blockSignals(False)
            else:
                self._spin_min.blockSignals(True)
                self._spin_min.setValue(self._spin_max.value())
                self._spin_min.blockSignals(False)
        self._emit()

    def _on_direction_toggled(self, _checked: bool) -> None:
        # Mutual exclusion: toggling one button unchecks the other
        sender = self.sender()
        if sender is self._btn_long and self._btn_long.isChecked():
            self._btn_short.blockSignals(True)
            self._btn_short.setChecked(False)
            self._btn_short.blockSignals(False)
        elif sender is self._btn_short and self._btn_short.isChecked():
            self._btn_long.blockSignals(True)
            self._btn_long.setChecked(False)
            self._btn_long.blockSignals(False)
        self._emit()

    def _on_menu_triggered(self, _action: QtGui.QAction) -> None:
        self._update_button_labels()
        self._emit()

    def _on_text_changed(self, _text: str) -> None:
        self._emit()
