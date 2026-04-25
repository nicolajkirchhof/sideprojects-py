"""
finance.apps.assistant._filter_proxy
======================================
Filter state and proxy model for the watchlist table — TA-E4-S2.

FilterState
-----------
Immutable dataclass describing the active filter. All fields default to
"no filter" so that FilterState() is always the identity (passes every row).

row_passes_filter
-----------------
Pure function — applies FilterState to a single result row dict.
Extracted from the proxy so it can be unit-tested without Qt.

WatchlistFilterProxy
--------------------
QSortFilterProxyModel subclass. Delegates accept/reject decisions entirely
to row_passes_filter(). Sorting behaviour inherited from the base class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pyqtgraph.Qt import QtCore

# ---------------------------------------------------------------------------
# FilterState
# ---------------------------------------------------------------------------

SCORE_MIN_DEFAULT: float = 0.0
# 100 base + 12 tag-bonus cap + 3 rounding headroom
SCORE_MAX_DEFAULT: float = 115.0
_DIRECTION_DEFAULT: str = "all"


@dataclass(frozen=True)
class FilterState:
    """
    Immutable snapshot of all active filter criteria.

    Tag and sector filters use OR semantics within their dimension: a row
    passes if it matches ANY selected tag / sector.  All dimensions are
    combined with AND (score AND direction AND tags AND sectors AND text).
    """

    score_min: float = SCORE_MIN_DEFAULT
    score_max: float = SCORE_MAX_DEFAULT
    direction: str = _DIRECTION_DEFAULT          # "all" | "long" | "short"
    tags: frozenset[str] = field(default_factory=frozenset)
    sectors: frozenset[str] = field(default_factory=frozenset)
    text: str = ""

    def is_default(self) -> bool:
        """Return True when no filters are active (identity state)."""
        return (
            self.score_min == SCORE_MIN_DEFAULT
            and self.score_max == SCORE_MAX_DEFAULT
            and self.direction == _DIRECTION_DEFAULT
            and not self.tags
            and not self.sectors
            and not self.text
        )

    def active_summary(self) -> str:
        """
        Return a compact human-readable summary of active filters,
        or an empty string when is_default() is True.

        Example: "score ≥40  |  dir: long  |  tags: vol-spike  |  sector: Technology"
        """
        parts: list[str] = []

        if self.score_min != SCORE_MIN_DEFAULT or self.score_max != SCORE_MAX_DEFAULT:
            lo = int(self.score_min)
            hi = int(self.score_max)
            parts.append(f"score {lo}–{hi}")

        if self.direction != _DIRECTION_DEFAULT:
            parts.append(f"dir: {self.direction}")

        if self.tags:
            parts.append("tags: " + ", ".join(sorted(self.tags)))

        if self.sectors:
            parts.append("sectors: " + ", ".join(sorted(self.sectors)))

        if self.text:
            parts.append(f'search: "{self.text}"')

        return "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Pure filter function
# ---------------------------------------------------------------------------


def row_passes_filter(row: dict, state: FilterState) -> bool:
    """
    Return True if *row* satisfies every criterion in *state*.

    Dimensions are AND-combined.  Within tags and sectors the check is OR
    (any matching tag/sector is sufficient).
    """
    if state.is_default():
        return True

    # Score range
    score = float(row.get("score_total") or 0.0)
    if score < state.score_min or score > state.score_max:
        return False

    # Direction
    if state.direction != _DIRECTION_DEFAULT:
        if (row.get("direction") or "") != state.direction:
            return False

    # Tags — OR: at least one selected tag must appear in the row
    if state.tags:
        row_tags = set(row.get("tags") or [])
        if not state.tags.intersection(row_tags):
            return False

    # Sectors — OR: the row's sector must be one of the selected sectors
    if state.sectors:
        if (row.get("sector") or "") not in state.sectors:
            return False

    # Text — substring match against symbol or sector (case-insensitive)
    if state.text:
        needle = state.text.lower()
        sym = (row.get("symbol") or "").lower()
        sec = (row.get("sector") or "").lower()
        if needle not in sym and needle not in sec:
            return False

    return True


# ---------------------------------------------------------------------------
# Qt proxy
# ---------------------------------------------------------------------------


class WatchlistFilterProxy(QtCore.QSortFilterProxyModel):
    """
    Sort+filter proxy for WatchlistModel.

    Sorting behaviour is inherited from QSortFilterProxyModel using the
    UserRole values set by WatchlistModel.

    Filtering is delegated to row_passes_filter().  Call update_filter()
    to apply a new FilterState; the proxy invalidates its filter cache
    automatically.
    """

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._state: FilterState = FilterState()
        self.setSortRole(QtCore.Qt.ItemDataRole.UserRole)
        self.setSortCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)

    def update_filter(self, state: FilterState) -> None:
        """Apply *state* as the active filter and refresh the view."""
        self._state = state
        self.invalidateFilter()

    def filterAcceptsRow(
        self,
        source_row: int,
        source_parent: QtCore.QModelIndex,
    ) -> bool:
        source = self.sourceModel()
        if source is None:
            return True
        # Retrieve the raw row dict via _rows on WatchlistModel.
        # Using internal attribute is acceptable here — proxy and model are
        # tightly coupled by design.
        try:
            row = source._rows[source_row]  # type: ignore[attr-defined]
        except (AttributeError, IndexError):
            return True
        return row_passes_filter(row, self._state)
