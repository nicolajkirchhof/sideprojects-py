"""
Tests for forex_oco_bt and currency_momentum_bt.
TDD: written before implementation — all imports will fail until modules exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finance.intraday_pm.forex.backtests.forex_oco_bt import (
    _to_pips,
    PIP_SIZE,
)
from finance.intraday_pm.forex.backtests.currency_momentum_bt import (
    _invert_returns,
    _momentum_score,
    _rank_long_short,
    _rebalance_cost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _weekly_series(values: list[float], start: str = "2020-01-10") -> pd.Series:
    """Build a weekly Friday-close price series."""
    idx = pd.date_range(start, periods=len(values), freq="W-FRI")
    return pd.Series(values, index=idx)


# ---------------------------------------------------------------------------
# _to_pips
# ---------------------------------------------------------------------------
class TestToPips:
    def test_eurusd_one_pip(self):
        assert _to_pips(0.0001, PIP_SIZE["EURUSD"]) == pytest.approx(1.0)

    def test_eurusd_ten_pips(self):
        assert _to_pips(0.0010, PIP_SIZE["EURUSD"]) == pytest.approx(10.0)

    def test_usdjpy_one_pip(self):
        # USDJPY pip = 0.01
        assert _to_pips(0.01, PIP_SIZE["USDJPY"]) == pytest.approx(1.0)

    def test_usdjpy_fifty_pips(self):
        assert _to_pips(0.50, PIP_SIZE["USDJPY"]) == pytest.approx(50.0)

    def test_negative_move(self):
        assert _to_pips(-0.0003, PIP_SIZE["GBPUSD"]) == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# _invert_returns
# ---------------------------------------------------------------------------
class TestInvertReturns:
    def test_positive_becomes_negative(self):
        s = pd.Series([0.01, 0.02, -0.01])
        result = _invert_returns(s)
        expected = pd.Series([-0.01, -0.02, 0.01])
        pd.testing.assert_series_equal(result, expected)

    def test_zeros_stay_zero(self):
        s = pd.Series([0.0, 0.0])
        pd.testing.assert_series_equal(_invert_returns(s), s)

    def test_series_preserved(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="W-FRI")
        s = pd.Series([0.01, -0.02, 0.03], index=idx)
        result = _invert_returns(s)
        assert list(result.index) == list(s.index)


# ---------------------------------------------------------------------------
# _momentum_score
# ---------------------------------------------------------------------------
class TestMomentumScore:
    def _make_prices(self, n: int = 60) -> pd.Series:
        """Monotonically rising prices — should give positive momentum score."""
        idx = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
        return pd.Series(range(1, n + 1), index=idx, dtype=float)

    def test_positive_trend_gives_positive_score(self):
        prices = self._make_prices(60)
        scores = _momentum_score(prices, lookback=52, skip=4)
        # The last score should be positive — prices rose over 52-4 week window
        last = scores.dropna().iloc[-1]
        assert last > 0

    def test_negative_trend_gives_negative_score(self):
        idx = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
        prices = pd.Series(range(60, 0, -1), index=idx, dtype=float)
        scores = _momentum_score(prices, lookback=52, skip=4)
        last = scores.dropna().iloc[-1]
        assert last < 0

    def test_no_lookahead(self):
        """Score at time t must only use data up to t-skip weeks."""
        prices = self._make_prices(60)
        scores = _momentum_score(prices, lookback=52, skip=4)
        # First valid score requires at least lookback weeks of data
        first_valid_idx = scores.first_valid_index()
        assert first_valid_idx is not None
        pos = prices.index.get_loc(first_valid_idx)
        assert pos >= 52  # need at least 52 bars for a valid score

    def test_nan_before_enough_history(self):
        prices = self._make_prices(30)  # fewer than lookback=52
        scores = _momentum_score(prices, lookback=52, skip=4)
        assert scores.dropna().empty


# ---------------------------------------------------------------------------
# _rank_long_short
# ---------------------------------------------------------------------------
class TestRankLongShort:
    def _make_scores(self) -> pd.Series:
        return pd.Series({
            "EURUSD": 0.05,
            "GBPUSD": 0.03,
            "AUDUSD": -0.01,
            "CHFUSD": 0.04,
            "inv_USDJPY": -0.04,
            "inv_USDCAD": 0.01,
        })

    def test_top_n_are_long(self):
        scores = self._make_scores()
        long_pos, short_pos = _rank_long_short(scores, n=3)
        assert set(long_pos) == {"EURUSD", "CHFUSD", "GBPUSD"}

    def test_bottom_n_are_short(self):
        scores = self._make_scores()
        long_pos, short_pos = _rank_long_short(scores, n=3)
        assert set(short_pos) == {"inv_USDJPY", "AUDUSD", "inv_USDCAD"}

    def test_no_overlap(self):
        scores = self._make_scores()
        long_pos, short_pos = _rank_long_short(scores, n=3)
        assert set(long_pos).isdisjoint(set(short_pos))

    def test_tie_broken_alphabetically(self):
        scores = pd.Series({"AAA": 0.05, "BBB": 0.05, "CCC": 0.01, "DDD": 0.01})
        long_pos, short_pos = _rank_long_short(scores, n=2)
        # Ties broken by alphabetical order of symbol
        assert set(long_pos) == {"AAA", "BBB"}
        assert set(short_pos) == {"CCC", "DDD"}


# ---------------------------------------------------------------------------
# _rebalance_cost
# ---------------------------------------------------------------------------
class TestRebalanceCost:
    def test_no_change_zero_cost(self):
        pos = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": -1}
        assert _rebalance_cost(pos, pos, cost_per_trade=1.5) == pytest.approx(0.0)

    def test_full_turnover(self):
        prev = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": -1}
        new = {"CHFUSD": 1, "inv_USDJPY": 1, "inv_USDCAD": -1}
        # 6 pairs changed → 6 × 1.5 = 9.0
        assert _rebalance_cost(prev, new, cost_per_trade=1.5) == pytest.approx(9.0)

    def test_partial_turnover(self):
        prev = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": -1}
        new = {"EURUSD": 1, "GBPUSD": -1, "AUDUSD": -1}
        # GBPUSD changed direction → 1 × 1.5 = 1.5
        assert _rebalance_cost(prev, new, cost_per_trade=1.5) == pytest.approx(1.5)

    def test_first_entry_costs_all(self):
        prev: dict = {}
        new = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": -1}
        # 3 new positions → 3 × 1.5 = 4.5
        assert _rebalance_cost(prev, new, cost_per_trade=1.5) == pytest.approx(4.5)
