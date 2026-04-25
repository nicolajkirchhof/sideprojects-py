"""
finance.apps.assistant._rules
==============================
Position rules engine for Trade Management (TA-E7-S1).

Pure Python, no Qt.  Each rule is a function that accepts a Position and
optional regime context and returns an Alert or None.

evaluate_position() is the public entry point — it runs all applicable
rules and returns a list of Alerts sorted by severity (critical first).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from finance.apps.assistant._data import TrendStatus

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "warn": 1, "ok": 2}


@dataclass
class Position:
    symbol: str
    position_type: Literal["stock", "option"]
    direction: Literal["long", "short"]
    entry_price: float
    current_price: float
    initial_risk: float        # 1R in dollars (stop distance × shares, or cost for options)
    pnl_dollars: float
    days_held: int
    underlying_trend: "TrendStatus | None" = None  # for option rules


@dataclass
class Alert:
    severity: Literal["ok", "warn", "critical"]
    rule: str
    message: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pnl_r(pos: Position) -> float:
    """Return P&L expressed as a multiple of initial risk (1R)."""
    if pos.initial_risk == 0:
        return 0.0
    return pos.pnl_dollars / pos.initial_risk


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

def rule_scale_out_1_5r(pos: Position) -> Alert | None:
    """Warn when unrealised gain is in the 1.5–4R range — first profit-taking step.

    Upper bound at 4R prevents stacking with rule_scale_out_4r.
    """
    r = _pnl_r(pos)
    if 1.5 <= r < 4.0:
        return Alert(
            severity="warn",
            rule="scale_out_1_5r",
            message=f"Scale out 30–50% — at {r:.1f}R ({pos.pnl_dollars:+.0f}$)",
        )
    return None


def rule_stop_to_breakeven(pos: Position) -> Alert | None:
    """Warn when unrealised gain is in the 2–4R range — move stop to breakeven.

    Upper bound at 4R prevents stacking with rule_scale_out_4r.
    """
    r = _pnl_r(pos)
    if 2.0 <= r < 4.0:
        return Alert(
            severity="warn",
            rule="stop_to_breakeven",
            message=f"Move stop to breakeven — at {r:.1f}R ({pos.pnl_dollars:+.0f}$)",
        )
    return None


def rule_scale_out_4r(pos: Position) -> Alert | None:
    """Warn when unrealised gain reaches 4R — second profit-taking step.

    Playbook §07: '2nd take: >4R → Close 30% more. Trail stop on 5 SMA.'
    """
    if _pnl_r(pos) >= 4.0:
        r = _pnl_r(pos)
        return Alert(
            severity="warn",
            rule="scale_out_4r",
            message=f"Scale out 30% more + trail stop on 5 SMA — at {r:.1f}R ({pos.pnl_dollars:+.0f}$)",
        )
    return None


def rule_loser_held_too_long(pos: Position, max_days: int = 50) -> Alert | None:
    """Critical when a losing position exceeds the time stop (default 50 days).

    Playbook §07 time stop: 50 days elapsed with thesis not playing out.
    The strategy timeframe is 5–50 days; a loss at 50d means capital should
    be redeployed.
    """
    if pos.days_held > max_days and _pnl_r(pos) < 0:
        return Alert(
            severity="critical",
            rule="loser_held_too_long",
            message=(
                f"Time stop — loser held {pos.days_held}d (>{max_days}d) — "
                f"P&L {pos.pnl_dollars:+.0f}$ ({_pnl_r(pos):.1f}R). Redeploy capital."
            ),
        )
    return None


def rule_option_underlying_broke_20sma(pos: Position) -> Alert | None:
    """Warn when a long option's underlying breaks below its 20-day SMA.

    Playbook §07 exit signal: 'Daily close below 20 SMA on above-avg volume.'
    The 20d SMA is the primary individual stock exit trigger (not the 50d,
    which is the regime filter).
    """
    if pos.position_type != "option" or pos.direction != "long":
        return None
    ts = pos.underlying_trend
    if ts is not None and not ts.price_above_20:
        return Alert(
            severity="warn",
            rule="option_underlying_broke_20sma",
            message=f"Underlying {ts.symbol} broke below 20d SMA ({ts.sma_20:.1f}) — exit signal",
        )
    return None


def rule_option_regime_nogo(pos: Position, regime: str) -> Alert | None:
    """Warn when a long option is held during a NO-GO regime."""
    if pos.position_type != "option" or pos.direction != "long":
        return None
    if regime.upper() == "NO-GO":
        return Alert(
            severity="warn",
            rule="option_regime_nogo",
            message="Long option in NO-GO regime — consider closing",
        )
    return None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def evaluate_position(pos: Position, regime_status: str = "") -> list[Alert]:
    """
    Run all applicable rules for *pos* and return Alerts sorted by severity.

    Parameters
    ----------
    pos:
        The position to evaluate.
    regime_status:
        Current regime string (e.g. "GO", "CAUTION", "NO-GO").

    Returns
    -------
    list[Alert]
        Sorted critical → warn → ok; empty when no rules fire.
    """
    alerts: list[Alert] = []

    for rule_fn in (rule_scale_out_1_5r, rule_stop_to_breakeven, rule_scale_out_4r):
        result = rule_fn(pos)
        if result is not None:
            alerts.append(result)

    result = rule_loser_held_too_long(pos)
    if result is not None:
        alerts.append(result)

    result = rule_option_underlying_broke_20sma(pos)
    if result is not None:
        alerts.append(result)

    result = rule_option_regime_nogo(pos, regime_status)
    if result is not None:
        alerts.append(result)

    alerts.sort(key=lambda a: _SEVERITY_ORDER.get(a.severity, 99))
    return alerts
