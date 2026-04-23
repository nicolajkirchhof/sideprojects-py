"""
finance.apps.assistant._watchlist_model
=========================================
Qt table model for the scored candidate watchlist.

WatchlistModel wraps list[dict] (result rows from the pipeline cache) and
presents them in a 14-column table suitable for QTableView + QSortFilterProxyModel.

Columns
-------
  CHECK | SYMBOL | DIR | SCORE | D1 | D2 | D3 | D4 | D5 | TAGS | PRICE | 5D% | RVOL | SECTOR

Roles used
----------
  DisplayRole     — formatted string shown to the user
  UserRole        — raw numeric/string value used by QSortFilterProxyModel for sorting
  BackgroundRole  — QBrush for score colour-coding (green/amber/red)
  CheckStateRole  — checkbox state in the CHECK column
  TextAlignmentRole — right-align numeric columns
"""
from __future__ import annotations

from enum import IntEnum

from pyqtgraph.Qt import QtCore, QtGui

# ---------------------------------------------------------------------------
# Column enumeration
# ---------------------------------------------------------------------------

class Col(IntEnum):
    CHECK     = 0
    SYMBOL    = 1
    DIRECTION = 2
    SCORE     = 3
    D1        = 4
    D2        = 5
    D3        = 6
    D4        = 7
    D5        = 8
    TAGS      = 9
    PRICE     = 10
    CHANGE_5D = 11
    RVOL      = 12
    SECTOR    = 13


COLUMN_COUNT: int = len(Col)

COLUMN_HEADERS: dict[Col, str] = {
    Col.CHECK:     "",
    Col.SYMBOL:    "Symbol",
    Col.DIRECTION: "Dir",
    Col.SCORE:     "Score",
    Col.D1:        "D1",
    Col.D2:        "D2",
    Col.D3:        "D3",
    Col.D4:        "D4",
    Col.D5:        "D5",
    Col.TAGS:      "Tags",
    Col.PRICE:     "Price",
    Col.CHANGE_5D: "5D%",
    Col.RVOL:      "RVOL",
    Col.SECTOR:    "Sector",
}

# ---------------------------------------------------------------------------
# Score thresholds and colours
# ---------------------------------------------------------------------------

_SCORE_GREEN: float = 70.0
_SCORE_AMBER: float = 40.0

_COLOR_GREEN = "#1a4a1a"
_COLOR_AMBER = "#4a3a00"
_COLOR_RED   = "#4a1a1a"

# Sentinel returned from UserRole for missing numeric values — sorts to bottom
# when the proxy sorts descending (all real scores are ≥ 0).
_MISSING_SORT_VALUE: float = -999.0

_NUMERIC_COLS: frozenset[Col] = frozenset({
    Col.SCORE, Col.D1, Col.D2, Col.D3, Col.D4, Col.D5,
    Col.PRICE, Col.CHANGE_5D, Col.RVOL,
})

_ItemDataRole = QtCore.Qt.ItemDataRole
_ItemFlag     = QtCore.Qt.ItemFlag
_CheckState   = QtCore.Qt.CheckState
_Alignment    = QtCore.Qt.AlignmentFlag


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WatchlistModel(QtCore.QAbstractTableModel):
    """
    Table model for the scored candidate watchlist.

    Data is loaded via load_rows(). The model is read-only except for the
    CHECK column which supports toggling via CheckStateRole / setData().

    Sorting is delegated to QSortFilterProxyModel using UserRole values.
    """

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict] = []
        self._checked: set[int] = set()  # row indices that are checked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_rows(self, rows: list[dict]) -> None:
        """
        Replace the current data with *rows* and reset all checkbox state.

        Triggers a full model reset so any attached views refresh.
        """
        self.beginResetModel()
        self._rows = list(rows)
        self._checked = set()
        self.endResetModel()

    def checked_symbols(self) -> list[str]:
        """Return symbols of checked rows in their current (model) order.

        Note: indices in _checked are source-model row indices.  When a
        QSortFilterProxyModel is attached the proxy maps its own row numbers
        to source rows; callers must use checked_symbols() rather than
        inspecting _checked directly.
        """
        return [
            self._rows[i]["symbol"]
            for i in sorted(self._checked)
            if i < len(self._rows)
        ]

    def checked_count(self) -> int:
        """Return the number of currently checked rows."""
        return len(self._checked)

    def check_rows(self, indices: list[int]) -> None:
        """Check the given source-model row indices (adds to existing selection).

        Emits dataChanged for each newly checked row's CHECK cell.
        Out-of-range indices are silently ignored.
        """
        for i in indices:
            if 0 <= i < len(self._rows) and i not in self._checked:
                self._checked.add(i)
                idx = self.index(i, Col.CHECK)
                self.dataChanged.emit(idx, idx, [_ItemDataRole.CheckStateRole])

    def uncheck_all(self) -> None:
        """Clear all checked rows and emit dataChanged for each affected cell."""
        for i in list(self._checked):
            self._checked.discard(i)
            idx = self.index(i, Col.CHECK)
            self.dataChanged.emit(idx, idx, [_ItemDataRole.CheckStateRole])

    def check_top_n(self, n: int) -> int:
        """Check the top *n* rows by score_total (descending); returns actual count checked.

        Existing checked state is replaced (uncheck_all is called first).
        If *n* exceeds rowCount, all rows are checked.
        """
        self.uncheck_all()
        ranked = sorted(
            range(len(self._rows)),
            key=lambda i: float(self._rows[i].get("score_total") or 0.0),
            reverse=True,
        )
        to_check = ranked[:n]
        self.check_rows(to_check)
        return len(to_check)

    # ------------------------------------------------------------------
    # QAbstractTableModel interface
    # ------------------------------------------------------------------

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else COLUMN_COUNT

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = _ItemDataRole.DisplayRole,
    ) -> object:
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == _ItemDataRole.DisplayRole
        ):
            try:
                return COLUMN_HEADERS[Col(section)]
            except ValueError:
                return None
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return _ItemFlag.NoItemFlags
        base = _ItemFlag.ItemIsEnabled | _ItemFlag.ItemIsSelectable
        if index.column() == Col.CHECK:
            return base | _ItemFlag.ItemIsUserCheckable
        return base

    def data(self, index: QtCore.QModelIndex, role: int = _ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or index.row() >= len(self._rows):
            return None

        row = self._rows[index.row()]
        try:
            col = Col(index.column())
        except ValueError:
            return None

        if role == _ItemDataRole.CheckStateRole:
            if col == Col.CHECK:
                return _CheckState.Checked if index.row() in self._checked else _CheckState.Unchecked
            return None

        if role == _ItemDataRole.DisplayRole:
            return self._display_value(row, col)

        if role == _ItemDataRole.BackgroundRole:
            if col == Col.SCORE:
                return self._score_brush(float(row.get("score_total") or 0.0))
            return None

        if role == _ItemDataRole.UserRole:
            return self._sort_value(row, col)

        if role == _ItemDataRole.TextAlignmentRole:
            if col in _NUMERIC_COLS:
                return _Alignment.AlignRight | _Alignment.AlignVCenter

        return None

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: object,
        role: int = _ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False
        if role == _ItemDataRole.CheckStateRole and index.column() == Col.CHECK:
            row_idx = index.row()
            if value == _CheckState.Checked:
                self._checked.add(row_idx)
            else:
                self._checked.discard(row_idx)
            self.dataChanged.emit(index, index, [_ItemDataRole.CheckStateRole])
            return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _display_value(self, row: dict, col: Col) -> str | None:
        if col == Col.CHECK:
            return None
        if col == Col.SYMBOL:
            return row.get("symbol") or ""
        if col == Col.DIRECTION:
            d = row.get("direction") or ""
            return d[:1].upper()  # "L" or "S"
        if col == Col.SCORE:
            v = row.get("score_total")
            return f"{v:.1f}" if v is not None else ""
        if Col.D1 <= col <= Col.D5:
            v = _dim_score(row, col - Col.D1)
            return f"{v:.1f}"
        if col == Col.TAGS:
            return ", ".join(row.get("tags") or [])
        if col == Col.PRICE:
            v = row.get("price")
            return f"{v:.2f}" if v is not None else ""
        if col == Col.CHANGE_5D:
            v = row.get("change_5d_pct")
            return f"{v:+.1f}%" if v is not None else ""
        if col == Col.RVOL:
            v = row.get("rvol_20d")
            return f"{v:.1f}x" if v is not None else ""
        if col == Col.SECTOR:
            return row.get("sector") or ""
        return None

    def _sort_value(self, row: dict, col: Col) -> object:
        if col == Col.SYMBOL:
            return row.get("symbol") or ""
        if col == Col.DIRECTION:
            return row.get("direction") or ""
        if col == Col.SCORE:
            return float(row.get("score_total") or 0.0)
        if Col.D1 <= col <= Col.D5:
            return _dim_score(row, col - Col.D1)
        if col == Col.PRICE:
            v = row.get("price")
            return float(v) if v is not None else _MISSING_SORT_VALUE
        if col == Col.CHANGE_5D:
            v = row.get("change_5d_pct")
            return float(v) if v is not None else _MISSING_SORT_VALUE
        if col == Col.RVOL:
            v = row.get("rvol_20d")
            return float(v) if v is not None else _MISSING_SORT_VALUE
        if col == Col.SECTOR:
            return row.get("sector") or ""
        if col == Col.TAGS:
            return float(len(row.get("tags") or []))
        return None

    @staticmethod
    def _score_brush(score: float) -> QtGui.QBrush:
        if score >= _SCORE_GREEN:
            color = _COLOR_GREEN
        elif score >= _SCORE_AMBER:
            color = _COLOR_AMBER
        else:
            color = _COLOR_RED
        return QtGui.QBrush(QtGui.QColor(color))


# ---------------------------------------------------------------------------
# Module-level helper (used by both model and tests)
# ---------------------------------------------------------------------------

def _dim_score(row: dict, dim_idx: int) -> float:
    """Extract the weighted_score for dimension *dim_idx* from a result row."""
    dims = row.get("dimensions") or []
    if dim_idx < len(dims):
        return float(dims[dim_idx].get("weighted_score") or 0.0)
    return 0.0
