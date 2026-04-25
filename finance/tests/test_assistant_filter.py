"""
Tests for finance.apps.assistant._filter_proxy — TA-E4-S2.

Pure-function tests for row_passes_filter do not require a Qt display.
The proxy integration test requires one (gated by _has_display).
"""
from __future__ import annotations

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)


def _row(
    symbol: str = "AAPL",
    direction: str = "long",
    score: float = 75.0,
    tags: list[str] | None = None,
    sector: str = "Technology",
) -> dict:
    return {
        "symbol": symbol,
        "direction": direction,
        "score_total": score,
        "tags": tags if tags is not None else [],
        "sector": sector,
    }


# ---------------------------------------------------------------------------
# FilterState.is_default()
# ---------------------------------------------------------------------------


def test_is_default_true_when_unchanged():
    from finance.apps.assistant._filter_proxy import FilterState

    assert FilterState().is_default()


def test_is_default_false_when_score_min_changed():
    from finance.apps.assistant._filter_proxy import FilterState, SCORE_MIN_DEFAULT

    assert not FilterState(score_min=SCORE_MIN_DEFAULT + 10.0).is_default()


def test_is_default_false_when_direction_set():
    from finance.apps.assistant._filter_proxy import FilterState

    assert not FilterState(direction="long").is_default()


def test_is_default_false_when_tags_set():
    from finance.apps.assistant._filter_proxy import FilterState

    assert not FilterState(tags=frozenset({"vol-spike"})).is_default()


def test_is_default_false_when_sectors_set():
    from finance.apps.assistant._filter_proxy import FilterState

    assert not FilterState(sectors=frozenset({"Technology"})).is_default()


def test_is_default_false_when_text_set():
    from finance.apps.assistant._filter_proxy import FilterState

    assert not FilterState(text="AA").is_default()


# ---------------------------------------------------------------------------
# row_passes_filter — score range
# ---------------------------------------------------------------------------


def test_default_filter_passes_all():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState()
    assert row_passes_filter(_row(score=0.0), state)
    assert row_passes_filter(_row(score=50.0), state)
    assert row_passes_filter(_row(score=100.0), state)


def test_score_min_excludes_below():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(score_min=40.0)
    assert not row_passes_filter(_row(score=39.9), state)
    assert row_passes_filter(_row(score=40.0), state)
    assert row_passes_filter(_row(score=80.0), state)


def test_score_max_excludes_above():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(score_max=70.0)
    assert row_passes_filter(_row(score=70.0), state)
    assert not row_passes_filter(_row(score=70.1), state)


def test_score_range_both_bounds():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(score_min=40.0, score_max=70.0)
    assert not row_passes_filter(_row(score=39.0), state)
    assert row_passes_filter(_row(score=55.0), state)
    assert not row_passes_filter(_row(score=71.0), state)


# ---------------------------------------------------------------------------
# row_passes_filter — direction
# ---------------------------------------------------------------------------


def test_direction_all_passes_both():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(direction="all")
    assert row_passes_filter(_row(direction="long"), state)
    assert row_passes_filter(_row(direction="short"), state)


def test_direction_long_excludes_short():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(direction="long")
    assert row_passes_filter(_row(direction="long"), state)
    assert not row_passes_filter(_row(direction="short"), state)


def test_direction_short_excludes_long():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(direction="short")
    assert row_passes_filter(_row(direction="short"), state)
    assert not row_passes_filter(_row(direction="long"), state)


# ---------------------------------------------------------------------------
# row_passes_filter — tags (OR semantics)
# ---------------------------------------------------------------------------


def test_tag_filter_passes_row_with_any_selected_tag():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(tags=frozenset({"vol-spike", "pead-long"}))
    assert row_passes_filter(_row(tags=["vol-spike"]), state)
    assert row_passes_filter(_row(tags=["pead-long", "52w-high"]), state)


def test_tag_filter_excludes_row_with_no_matching_tag():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(tags=frozenset({"vol-spike"}))
    assert not row_passes_filter(_row(tags=["pead-long", "52w-high"]), state)
    assert not row_passes_filter(_row(tags=[]), state)


def test_empty_tag_filter_passes_all():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(tags=frozenset())
    assert row_passes_filter(_row(tags=[]), state)
    assert row_passes_filter(_row(tags=["vol-spike"]), state)


# ---------------------------------------------------------------------------
# row_passes_filter — sectors (OR semantics)
# ---------------------------------------------------------------------------


def test_sector_filter_passes_matching():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(sectors=frozenset({"Technology", "Healthcare"}))
    assert row_passes_filter(_row(sector="Technology"), state)
    assert row_passes_filter(_row(sector="Healthcare"), state)


def test_sector_filter_excludes_non_matching():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(sectors=frozenset({"Technology"}))
    assert not row_passes_filter(_row(sector="Consumer"), state)


def test_empty_sector_filter_passes_all():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(sectors=frozenset())
    assert row_passes_filter(_row(sector="Technology"), state)
    assert row_passes_filter(_row(sector="Energy"), state)


# ---------------------------------------------------------------------------
# row_passes_filter — text search
# ---------------------------------------------------------------------------


def test_text_search_matches_symbol_substring():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(text="AA")
    assert row_passes_filter(_row(symbol="AAPL"), state)
    assert not row_passes_filter(_row(symbol="TSLA"), state)


def test_text_search_matches_sector_substring():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(text="tech")
    assert row_passes_filter(_row(symbol="MSFT", sector="Technology"), state)
    assert not row_passes_filter(_row(symbol="MSFT", sector="Consumer"), state)


def test_text_search_is_case_insensitive():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(text="aapl")
    assert row_passes_filter(_row(symbol="AAPL"), state)


def test_empty_text_passes_all():
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(text="")
    assert row_passes_filter(_row(symbol="AAPL"), state)
    assert row_passes_filter(_row(symbol="TSLA"), state)


# ---------------------------------------------------------------------------
# row_passes_filter — AND combination
# ---------------------------------------------------------------------------


def test_filters_are_and_combined():
    """A row must pass ALL active filters to be included."""
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter

    state = FilterState(
        score_min=50.0,
        direction="long",
        tags=frozenset({"vol-spike"}),
    )

    # Passes all three
    assert row_passes_filter(_row(score=75.0, direction="long", tags=["vol-spike"]), state)

    # Fails score
    assert not row_passes_filter(_row(score=40.0, direction="long", tags=["vol-spike"]), state)

    # Fails direction
    assert not row_passes_filter(_row(score=75.0, direction="short", tags=["vol-spike"]), state)

    # Fails tag
    assert not row_passes_filter(_row(score=75.0, direction="long", tags=["52w-high"]), state)


# ---------------------------------------------------------------------------
# WatchlistFilterProxy — Qt integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_proxy_row_count_respects_filter():
    """Proxy must expose only rows that pass the active FilterState."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._filter_proxy import FilterState, WatchlistFilterProxy
    from finance.apps.assistant._watchlist_model import WatchlistModel

    ensure_qt_app()

    rows = [
        _row(symbol="AAPL", direction="long", score=75.0),
        _row(symbol="TSLA", direction="short", score=35.0),
        _row(symbol="MSFT", direction="long", score=55.0),
    ]

    source = WatchlistModel()
    source.load_rows(rows)

    proxy = WatchlistFilterProxy()
    proxy.setSourceModel(source)

    assert proxy.rowCount() == 3

    proxy.update_filter(FilterState(direction="long"))
    assert proxy.rowCount() == 2

    proxy.update_filter(FilterState(score_min=60.0))
    assert proxy.rowCount() == 1  # only AAPL

    proxy.update_filter(FilterState())
    assert proxy.rowCount() == 3


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_proxy_filter_by_tag():
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._filter_proxy import FilterState, WatchlistFilterProxy
    from finance.apps.assistant._watchlist_model import WatchlistModel

    ensure_qt_app()

    rows = [
        _row(symbol="AAPL", tags=["vol-spike", "52w-high"]),
        _row(symbol="TSLA", tags=["pead-short"]),
        _row(symbol="MSFT", tags=[]),
    ]

    source = WatchlistModel()
    source.load_rows(rows)

    proxy = WatchlistFilterProxy()
    proxy.setSourceModel(source)

    proxy.update_filter(FilterState(tags=frozenset({"vol-spike"})))
    assert proxy.rowCount() == 1


# ---------------------------------------------------------------------------
# Tag-bonus score > 100 — default filter must not hide bonus rows
# ---------------------------------------------------------------------------


def test_default_filter_passes_score_above_100():
    """Rows with tag bonuses can score above 100; default max (115) must include them."""
    from finance.apps.assistant._filter_proxy import FilterState, row_passes_filter, SCORE_MAX_DEFAULT

    assert SCORE_MAX_DEFAULT >= 112  # 100 base + 12 tag-bonus cap
    state = FilterState()
    assert row_passes_filter(_row(score=104.0), state)
    assert row_passes_filter(_row(score=112.0), state)


def test_score_max_115_is_default_identity():
    """FilterState() with score_max=115 must report is_default() True."""
    from finance.apps.assistant._filter_proxy import FilterState, SCORE_MAX_DEFAULT

    assert FilterState(score_max=SCORE_MAX_DEFAULT).is_default()


# ---------------------------------------------------------------------------
# update_options emits so proxy state is reset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_update_options_emits_reset_filter():
    """update_options() must emit filters_changed so the proxy sees cleared selections."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._filter_bar import FilterBar
    from finance.apps.assistant._filter_proxy import FilterState

    ensure_qt_app()

    bar = FilterBar()
    emitted: list[FilterState] = []
    bar.filters_changed.connect(emitted.append)

    # Populate with some options
    bar.update_options({"vol-spike": 3, "pead-long": 1}, {"Technology": 5})

    # update_options should have emitted once (the reset emit)
    assert len(emitted) >= 1
    last = emitted[-1]
    assert last.is_default() or (not last.tags and not last.sectors)
