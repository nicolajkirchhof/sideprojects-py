"""
Tests for OCO simulation functions in hougaard_dax.py.

Covers the _fixed_2r exit mode added to support ORB-style trade management
(hard stop at opposite bracket side + fixed 2R take-profit target).
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch, call

import numpy as np
import pandas as pd
import pytest

from finance.intraday_pm.backtests.hougaard_dax import (
    _fixed_2r,
    _simulate_oco,
    ENTRY_OFFSET_PTS,
    SPREAD_COST_PTS,
)
from finance.intraday_pm.backtests.candle_scan import _scan_one

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(ohlc_list: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a minimal bars DataFrame from (open, high, low, close) tuples."""
    base = pd.Timestamp("2024-01-02 10:00:00", tz="UTC")
    if not ohlc_list:
        return pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
    idx = pd.date_range(base, periods=len(ohlc_list), freq="5min")
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in ohlc_list],
        index=idx,
    )


def _signal_bar(high: float, low: float) -> pd.Series:
    """Create a minimal signal bar with a timestamp index."""
    ts = pd.Timestamp("2024-01-02 09:45:00", tz="UTC")
    mid = (high + low) / 2
    return pd.Series({"open": mid, "high": high, "low": low, "close": mid}, name=ts)


# ---------------------------------------------------------------------------
# _fixed_2r unit tests
# Scenario: long entry=102, stop=88, risk=14, target=130
#           short entry=98,  stop=112, risk=14, target=70
# ---------------------------------------------------------------------------

LONG_ENTRY  = 102.0
LONG_STOP   = 88.0
SHORT_ENTRY = 98.0
SHORT_STOP  = 112.0
RISK        = 14.0
LONG_TARGET  = LONG_ENTRY  + 2 * RISK   # 130
SHORT_TARGET = SHORT_ENTRY - 2 * RISK   # 70


class TestFixed2r:
    def test_long_target_hit(self):
        bars = _make_bars([(101, 135, 100, 125)])
        _, exit_price = _fixed_2r("long", LONG_STOP, LONG_ENTRY, bars)
        assert exit_price == LONG_TARGET

    def test_long_stop_hit(self):
        bars = _make_bars([(101, 103, 85, 90)])
        stop_val, exit_price = _fixed_2r("long", LONG_STOP, LONG_ENTRY, bars)
        assert exit_price == LONG_STOP
        assert stop_val == LONG_STOP

    def test_long_eod_exit(self):
        bars = _make_bars([(103, 110, 100, 108)])  # high < 130, low > 88
        _, exit_price = _fixed_2r("long", LONG_STOP, LONG_ENTRY, bars)
        assert exit_price == 108.0  # last close

    def test_long_empty_bars_exits_at_entry(self):
        bars = _make_bars([])
        _, exit_price = _fixed_2r("long", LONG_STOP, LONG_ENTRY, bars)
        assert exit_price == LONG_ENTRY

    def test_long_stop_takes_priority_when_same_bar_hits_both(self):
        """Conservative: stop checked before target within the same bar."""
        bars = _make_bars([(101, 135, 85, 110)])  # low<=88 AND high>=130
        _, exit_price = _fixed_2r("long", LONG_STOP, LONG_ENTRY, bars)
        assert exit_price == LONG_STOP

    def test_short_target_hit(self):
        bars = _make_bars([(97, 99, 65, 72)])
        _, exit_price = _fixed_2r("short", SHORT_STOP, SHORT_ENTRY, bars)
        assert exit_price == SHORT_TARGET

    def test_short_stop_hit(self):
        bars = _make_bars([(99, 115, 97, 113)])
        stop_val, exit_price = _fixed_2r("short", SHORT_STOP, SHORT_ENTRY, bars)
        assert exit_price == SHORT_STOP
        assert stop_val == SHORT_STOP

    def test_short_eod_exit(self):
        bars = _make_bars([(98, 100, 95, 96)])  # high < 112, low > 70
        _, exit_price = _fixed_2r("short", SHORT_STOP, SHORT_ENTRY, bars)
        assert exit_price == 96.0

    def test_short_stop_takes_priority_when_same_bar_hits_both(self):
        bars = _make_bars([(98, 115, 65, 90)])  # high>=112 AND low<=70
        _, exit_price = _fixed_2r("short", SHORT_STOP, SHORT_ENTRY, bars)
        assert exit_price == SHORT_STOP

    def test_multi_bar_target_on_second_bar(self):
        """Target not hit on first bar, hit on second."""
        bars = _make_bars([
            (103, 110, 100, 107),  # no hit
            (108, 135, 107, 130),  # target hit
        ])
        _, exit_price = _fixed_2r("long", LONG_STOP, LONG_ENTRY, bars)
        assert exit_price == LONG_TARGET


# ---------------------------------------------------------------------------
# _simulate_oco integration tests with exit_mode="fixed_2r"
# Signal bar: high=100, low=90 → range=10
# stop_pts=14 → long: entry=102, stop=88, target=130
#                short: entry=88, stop=102, target=60
# ---------------------------------------------------------------------------

class TestSimulateOcoFixed2r:
    STOP_PTS = 14.0  # bar_range(10) + 2*offset(4)

    def test_long_fill_target_hit(self):
        signal = _signal_bar(high=100.0, low=90.0)
        remaining = _make_bars([
            (101, 103, 99, 102),   # fills long at 102
            (103, 135, 102, 128),  # target 130 hit
        ])
        rec = _simulate_oco(signal, remaining, self.STOP_PTS, exit_mode="fixed_2r")
        assert rec["filled_direction"] == "long"
        expected = (LONG_TARGET - LONG_ENTRY) - SPREAD_COST_PTS  # +26
        assert rec["result_pts"] == pytest.approx(expected)
        assert rec["win"] is True

    def test_long_fill_stop_hit(self):
        signal = _signal_bar(high=100.0, low=90.0)
        remaining = _make_bars([
            (101, 103, 99, 102),  # fills long at 102
            (101, 103, 85, 90),   # stop at 88 hit
        ])
        rec = _simulate_oco(signal, remaining, self.STOP_PTS, exit_mode="fixed_2r")
        assert rec["filled_direction"] == "long"
        expected = (LONG_STOP - LONG_ENTRY) - SPREAD_COST_PTS  # -16
        assert rec["result_pts"] == pytest.approx(expected)
        assert rec["win"] is False

    def test_no_fill_returns_nan(self):
        signal = _signal_bar(high=100.0, low=90.0)
        remaining = _make_bars([
            (95, 101, 89, 95),  # high<102 and low>88 — no fill
        ])
        rec = _simulate_oco(signal, remaining, self.STOP_PTS, exit_mode="fixed_2r")
        assert rec["filled_direction"] is None
        assert np.isnan(rec["result_pts"])

    def test_2bar_trail_mode_regression(self):
        """Renaming use_2bar_trail → exit_mode must not break existing trailing behaviour."""
        signal = _signal_bar(high=100.0, low=90.0)
        remaining = _make_bars([
            (101, 103, 99, 102),   # fills long at 102
            (103, 115, 101, 110),
            (110, 120, 109, 115),
        ])
        rec = _simulate_oco(signal, remaining, self.STOP_PTS, exit_mode="2bar_trail")
        assert rec["filled_direction"] == "long"
        assert not np.isnan(rec["result_pts"])

    def test_unknown_exit_mode_raises(self):
        signal = _signal_bar(high=100.0, low=90.0)
        remaining = _make_bars([(101, 103, 99, 102), (103, 135, 102, 128)])
        with pytest.raises(ValueError, match="Unknown exit_mode"):
            _simulate_oco(signal, remaining, self.STOP_PTS, exit_mode="bad_mode")


# ---------------------------------------------------------------------------
# _scan_one routing tests
# ---------------------------------------------------------------------------

def _make_sessions() -> dict[date, pd.DataFrame]:
    """Two minimal sessions with enough bars to use bar_idx=0."""
    base = pd.Timestamp("2024-01-02 09:00:00", tz="Europe/Berlin")
    rows = [
        {"open": 100, "high": 105, "low": 95, "close": 102},
        {"open": 102, "high": 115, "low": 95, "close": 110},
        {"open": 110, "high": 120, "low": 108, "close": 115},
    ]
    idx = pd.date_range(base, periods=len(rows), freq="5min")
    df = pd.DataFrame(rows, index=idx)
    return {base.date(): df}


class TestScanOneRouting:
    """Verify that _scan_one passes the correct exit_mode to _simulate_oco."""

    _DUMMY_RESULT = {
        "filled_direction": "long",
        "result_pts": 5.0,
        "entry": 107.0,
        "win": True,
        "bar_range": 10.0,
    }

    def _captured_exit_modes(self, stop_method: str) -> list[str]:
        sessions = _make_sessions()
        atr = pd.Series({list(sessions.keys())[0]: 50.0})
        captured: list[str] = []

        def fake_simulate(signal_bar, remaining, stop_pts, exit_mode="2bar_trail", atr_pts=np.nan):
            captured.append(exit_mode)
            return self._DUMMY_RESULT.copy()

        with patch(
            "finance.intraday_pm.backtests.candle_scan._simulate_oco",
            side_effect=fake_simulate,
        ):
            _scan_one(sessions, atr, bar_idx=0, stop_method=stop_method)

        return captured

    def test_bar_range_uses_2bar_trail(self):
        modes = self._captured_exit_modes("bar_range")
        assert modes and all(m == "2bar_trail" for m in modes)

    def test_atr_uses_2bar_trail(self):
        modes = self._captured_exit_modes("atr")
        assert modes and all(m == "2bar_trail" for m in modes)

    def test_fixed_2r_uses_fixed_2r(self):
        modes = self._captured_exit_modes("fixed_2r")
        assert modes and all(m == "fixed_2r" for m in modes)

    def test_unknown_stop_method_raises(self):
        sessions = _make_sessions()
        atr = pd.Series({list(sessions.keys())[0]: 50.0})
        with pytest.raises(ValueError, match="Unknown stop_method"):
            _scan_one(sessions, atr, bar_idx=0, stop_method="bad_method")
