"""
Tests for TA-E7-S1 — Position rules engine.

All tests are Qt-free and operate on pure-Python dataclasses.
"""
from __future__ import annotations

import pytest

from finance.apps.assistant._rules import (
    Alert,
    Position,
    evaluate_position,
    rule_loser_held_too_long,
    rule_option_regime_nogo,
    rule_option_underlying_broke_20sma,
    rule_scale_out_1_5r,
    rule_scale_out_4r,
    rule_stop_to_breakeven,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stock_pos(
    *,
    pnl_dollars: float = 0.0,
    initial_risk: float = 100.0,
    days_held: int = 3,
    direction: str = "long",
) -> Position:
    return Position(
        symbol="AAPL",
        position_type="stock",
        direction=direction,  # type: ignore[arg-type]
        entry_price=100.0,
        current_price=100.0 + pnl_dollars,
        initial_risk=initial_risk,
        pnl_dollars=pnl_dollars,
        days_held=days_held,
    )


def _option_pos(
    *,
    pnl_dollars: float = 0.0,
    initial_risk: float = 200.0,
    days_held: int = 5,
    direction: str = "long",
    underlying_trend=None,
) -> Position:
    return Position(
        symbol="AAPL",
        position_type="option",
        direction=direction,  # type: ignore[arg-type]
        entry_price=2.0,
        current_price=2.0 + pnl_dollars / 100,
        initial_risk=initial_risk,
        pnl_dollars=pnl_dollars,
        days_held=days_held,
        underlying_trend=underlying_trend,
    )


# ---------------------------------------------------------------------------
# rule_scale_out_1_5r
# ---------------------------------------------------------------------------

def test_scale_out_alert_fires_at_1_5r():
    pos = _stock_pos(pnl_dollars=150.0, initial_risk=100.0)  # exactly 1.5R
    alert = rule_scale_out_1_5r(pos)
    assert alert is not None
    assert alert.severity == "warn"
    assert "Scale out 30–50%" in alert.message
    assert alert.rule == "scale_out_1_5r"


def test_scale_out_alert_fires_above_1_5r():
    pos = _stock_pos(pnl_dollars=200.0, initial_risk=100.0)  # 2R
    alert = rule_scale_out_1_5r(pos)
    assert alert is not None


def test_scale_out_alert_absent_below_threshold():
    pos = _stock_pos(pnl_dollars=140.0, initial_risk=100.0)  # 1.4R
    alert = rule_scale_out_1_5r(pos)
    assert alert is None


def test_scale_out_alert_absent_at_zero():
    pos = _stock_pos(pnl_dollars=0.0, initial_risk=100.0)
    alert = rule_scale_out_1_5r(pos)
    assert alert is None


def test_scale_out_1_5r_absent_at_4r_and_above():
    """Upper-bound guard: 1.5R rule must not stack with rule_scale_out_4r."""
    pos = _stock_pos(pnl_dollars=400.0, initial_risk=100.0)  # exactly 4R
    assert rule_scale_out_1_5r(pos) is None

    pos2 = _stock_pos(pnl_dollars=500.0, initial_risk=100.0)  # 5R
    assert rule_scale_out_1_5r(pos2) is None


# ---------------------------------------------------------------------------
# rule_stop_to_breakeven
# ---------------------------------------------------------------------------

def test_stop_to_breakeven_fires_at_2r():
    pos = _stock_pos(pnl_dollars=200.0, initial_risk=100.0)
    alert = rule_stop_to_breakeven(pos)
    assert alert is not None
    assert alert.severity == "warn"
    assert "breakeven" in alert.message.lower()


def test_stop_to_breakeven_absent_below_2r():
    pos = _stock_pos(pnl_dollars=190.0, initial_risk=100.0)  # 1.9R
    alert = rule_stop_to_breakeven(pos)
    assert alert is None


def test_stop_to_breakeven_absent_at_4r_and_above():
    """Upper-bound guard: breakeven rule must not stack with rule_scale_out_4r."""
    pos = _stock_pos(pnl_dollars=400.0, initial_risk=100.0)  # exactly 4R
    assert rule_stop_to_breakeven(pos) is None

    pos2 = _stock_pos(pnl_dollars=500.0, initial_risk=100.0)  # 5R
    assert rule_stop_to_breakeven(pos2) is None


# ---------------------------------------------------------------------------
# rule_loser_held_too_long
# ---------------------------------------------------------------------------

def test_loser_held_too_long_fires():
    pos = _stock_pos(pnl_dollars=-50.0, days_held=51, initial_risk=100.0)
    alert = rule_loser_held_too_long(pos)  # default max_days=50
    assert alert is not None
    assert alert.severity == "critical"
    assert "51d" in alert.message
    assert alert.rule == "loser_held_too_long"


def test_loser_held_too_long_default_is_50_days():
    """Default time stop is 50 days per playbook §07."""
    pos = _stock_pos(pnl_dollars=-50.0, days_held=51, initial_risk=100.0)
    alert = rule_loser_held_too_long(pos)
    assert alert is not None  # fires after 50d

    pos_within = _stock_pos(pnl_dollars=-50.0, days_held=50, initial_risk=100.0)
    assert rule_loser_held_too_long(pos_within) is None  # strictly >


def test_loser_held_too_long_absent_for_winner():
    pos = _stock_pos(pnl_dollars=50.0, days_held=55, initial_risk=100.0)
    alert = rule_loser_held_too_long(pos)
    assert alert is None


def test_loser_held_too_long_absent_within_limit():
    pos = _stock_pos(pnl_dollars=-50.0, days_held=49, initial_risk=100.0)
    alert = rule_loser_held_too_long(pos)
    assert alert is None


def test_loser_held_too_long_absent_exactly_at_limit():
    pos = _stock_pos(pnl_dollars=-50.0, days_held=50, initial_risk=100.0)
    alert = rule_loser_held_too_long(pos)
    assert alert is None  # strictly >


def test_loser_held_too_long_custom_max_days():
    pos = _stock_pos(pnl_dollars=-50.0, days_held=11, initial_risk=100.0)
    alert = rule_loser_held_too_long(pos, max_days=10)
    assert alert is not None
    assert "11d" in alert.message


# ---------------------------------------------------------------------------
# rule_scale_out_4r
# ---------------------------------------------------------------------------

def test_scale_out_4r_fires():
    pos = _stock_pos(pnl_dollars=400.0, initial_risk=100.0)  # exactly 4R
    alert = rule_scale_out_4r(pos)
    assert alert is not None
    assert alert.severity == "warn"
    assert "30%" in alert.message
    assert "5 SMA" in alert.message
    assert alert.rule == "scale_out_4r"


def test_scale_out_4r_fires_above_4r():
    pos = _stock_pos(pnl_dollars=500.0, initial_risk=100.0)  # 5R
    alert = rule_scale_out_4r(pos)
    assert alert is not None


def test_scale_out_4r_absent_below_4r():
    pos = _stock_pos(pnl_dollars=390.0, initial_risk=100.0)  # 3.9R
    alert = rule_scale_out_4r(pos)
    assert alert is None


def test_scale_out_4r_absent_at_zero():
    pos = _stock_pos(pnl_dollars=0.0, initial_risk=100.0)
    alert = rule_scale_out_4r(pos)
    assert alert is None


# ---------------------------------------------------------------------------
# rule_option_underlying_broke_20sma
# ---------------------------------------------------------------------------

class _FakeTrendStatus:
    def __init__(
        self,
        *,
        symbol: str = "SPY",
        price_above_20: bool = True,
        sma_20: float = 410.0,
        price_above_50: bool = True,
        sma_50: float = 400.0,
    ):
        self.symbol = symbol
        self.price_above_20 = price_above_20
        self.sma_20 = sma_20
        self.price_above_50 = price_above_50
        self.sma_50 = sma_50


def test_option_underlying_broke_20sma_fires():
    trend = _FakeTrendStatus(symbol="SPY", price_above_20=False, sma_20=420.0)
    pos = _option_pos(underlying_trend=trend)
    alert = rule_option_underlying_broke_20sma(pos)
    assert alert is not None
    assert alert.severity == "warn"
    assert "20d SMA" in alert.message
    assert alert.rule == "option_underlying_broke_20sma"


def test_option_underlying_broke_20sma_absent_when_above():
    trend = _FakeTrendStatus(price_above_20=True)
    pos = _option_pos(underlying_trend=trend)
    alert = rule_option_underlying_broke_20sma(pos)
    assert alert is None


def test_option_underlying_broke_20sma_absent_for_stock():
    trend = _FakeTrendStatus(price_above_20=False)
    pos2 = Position(
        symbol="AAPL",
        position_type="stock",
        direction="long",
        entry_price=100.0,
        current_price=100.0,
        initial_risk=100.0,
        pnl_dollars=0.0,
        days_held=1,
        underlying_trend=trend,
    )
    alert = rule_option_underlying_broke_20sma(pos2)
    assert alert is None


def test_option_underlying_broke_20sma_absent_when_no_trend():
    pos = _option_pos(underlying_trend=None)
    alert = rule_option_underlying_broke_20sma(pos)
    assert alert is None


def test_option_underlying_broke_20sma_absent_for_short_option():
    trend = _FakeTrendStatus(price_above_20=False)
    pos = _option_pos(direction="short", underlying_trend=trend)
    alert = rule_option_underlying_broke_20sma(pos)
    assert alert is None


# ---------------------------------------------------------------------------
# rule_option_regime_nogo
# ---------------------------------------------------------------------------

def test_option_regime_nogo_fires():
    pos = _option_pos()
    alert = rule_option_regime_nogo(pos, "NO-GO")
    assert alert is not None
    assert alert.severity == "warn"
    assert "NO-GO" in alert.message
    assert alert.rule == "option_regime_nogo"


def test_option_regime_nogo_case_insensitive():
    pos = _option_pos()
    alert = rule_option_regime_nogo(pos, "no-go")
    assert alert is not None


def test_option_regime_nogo_absent_for_go():
    pos = _option_pos()
    alert = rule_option_regime_nogo(pos, "GO")
    assert alert is None


def test_option_regime_nogo_absent_for_stock():
    pos = _stock_pos()
    alert = rule_option_regime_nogo(pos, "NO-GO")
    assert alert is None


def test_option_regime_nogo_absent_for_short_option():
    pos = _option_pos(direction="short")
    alert = rule_option_regime_nogo(pos, "NO-GO")
    assert alert is None


# ---------------------------------------------------------------------------
# evaluate_position
# ---------------------------------------------------------------------------

def test_evaluate_position_returns_sorted_alerts():
    """Critical alerts must appear before warn alerts."""
    trend = _FakeTrendStatus(price_above_20=False)
    pos = Position(
        symbol="AAPL",
        position_type="option",
        direction="long",
        entry_price=2.0,
        current_price=1.0,
        initial_risk=200.0,
        pnl_dollars=-100.0,
        days_held=51,   # triggers loser_held_too_long (critical, default 50d)
        underlying_trend=trend,  # triggers option_underlying_broke_20sma (warn)
    )
    alerts = evaluate_position(pos, "NO-GO")
    assert len(alerts) >= 2
    severities = [a.severity for a in alerts]
    # critical must come before warn
    if "critical" in severities and "warn" in severities:
        assert severities.index("critical") < severities.index("warn")


def test_evaluate_position_empty_when_no_rules_fire():
    pos = _stock_pos(pnl_dollars=0.0, days_held=1, initial_risk=100.0)
    alerts = evaluate_position(pos, "GO")
    assert alerts == []


def test_evaluate_position_returns_list():
    pos = _stock_pos(pnl_dollars=200.0, initial_risk=100.0)
    alerts = evaluate_position(pos)
    assert isinstance(alerts, list)
    assert all(isinstance(a, Alert) for a in alerts)


def test_evaluate_position_zero_initial_risk_no_crash():
    pos = _stock_pos(pnl_dollars=100.0, initial_risk=0.0)
    alerts = evaluate_position(pos)
    assert isinstance(alerts, list)
