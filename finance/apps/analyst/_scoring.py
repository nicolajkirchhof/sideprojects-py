"""
finance.apps.analyst._scoring
================================
5-box checklist scoring for swing trade candidates.

Box 1: Trend Template (Minervini)
Box 2: Relative Strength / Weakness (Bruzzese)
Box 3: Base Quality (Minervini + Kell)
Box 4: Catalyst (manual / Claude)
Box 5: Risk Parameters
"""
from __future__ import annotations

import logging

from finance.apps.analyst._models import (
    BoxResult,
    EnrichedCandidate,
    ScoredCandidate,
    TechnicalData,
)

log = logging.getLogger(__name__)

MAX_STOP_PCT = 7.0       # Minervini hard limit: stop ≤ 7% from entry
MAX_52W_DISTANCE = 25.0  # Price within 25% of 52W high


def score(candidates: list[EnrichedCandidate]) -> list[ScoredCandidate]:
    """Score all candidates against the 5-box checklist, sorted by score then RS."""
    results = [_score_one(c) for c in candidates]
    results.sort(key=_sort_key, reverse=True)
    return results


def _score_one(ec: EnrichedCandidate) -> ScoredCandidate:
    """Score a single candidate against all 5 boxes."""
    t = ec.technicals
    if not ec.data_available or t is None:
        boxes = [
            BoxResult(1, "Trend Template", "MANUAL", "No IBKR data available"),
            BoxResult(2, "RS/RW", "MANUAL", "No IBKR data available"),
            BoxResult(3, "Base Quality", "MANUAL", "No IBKR data available"),
            BoxResult(4, "Catalyst", "MANUAL", "Requires qualitative assessment"),
            BoxResult(5, "Risk", "MANUAL", "No IBKR data available"),
        ]
        return ScoredCandidate(enriched=ec, boxes=boxes, score=0)

    boxes = [
        _box1_trend_template(t, ec.candidate.price),
        _box2_relative_strength(t),
        _box3_base_quality(t),
        _box4_catalyst(),
        _box5_risk(t, ec.candidate.price),
    ]
    passed = sum(1 for b in boxes if b.status == "PASS")
    return ScoredCandidate(enriched=ec, boxes=boxes, score=passed)


def _box1_trend_template(t: TechnicalData, price: float | None) -> BoxResult:
    """Minervini Trend Template: Price > 20 SMA > 50 SMA, 50 SMA rising,
    within 25% of 52W high, positive 12-month return."""
    if price is None or price <= 0:
        return BoxResult(1, "Trend Template", "MANUAL", "Price not available")

    reasons: list[str] = []
    fails: list[str] = []

    p = price
    if t.sma_20 is not None and t.sma_50 is not None:
        if p > t.sma_20 > t.sma_50:
            reasons.append(f"Price {p:.2f} > 20 SMA {t.sma_20:.2f} > 50 SMA {t.sma_50:.2f}")
        else:
            fails.append(f"SMA stack broken: Price {p:.2f}, 20 SMA {t.sma_20:.2f}, 50 SMA {t.sma_50:.2f}")
    else:
        fails.append("SMA data insufficient")

    if t.sma_50_slope == "rising":
        reasons.append("50 SMA rising")
    elif t.sma_50_slope is not None:
        fails.append(f"50 SMA {t.sma_50_slope}")

    if t.high_52w is not None and p > 0:
        distance = (t.high_52w - p) / t.high_52w * 100
        if distance <= MAX_52W_DISTANCE:
            reasons.append(f"Within {distance:.1f}% of 52W high")
        else:
            fails.append(f"{distance:.1f}% below 52W high (max {MAX_52W_DISTANCE}%)")

    if t.return_12m is not None:
        if t.return_12m > 0:
            reasons.append(f"12M return +{t.return_12m:.1f}%")
        else:
            fails.append(f"12M return {t.return_12m:.1f}% (negative)")

    if fails:
        return BoxResult(1, "Trend Template", "FAIL", "; ".join(fails))
    if not reasons:
        return BoxResult(1, "Trend Template", "MANUAL", "Insufficient data for evaluation")
    return BoxResult(1, "Trend Template", "PASS", "; ".join(reasons))


def _box2_relative_strength(t: TechnicalData) -> BoxResult:
    """RS line vs SPY trending up, outperforming over 1M."""
    if t.rs_slope_10d is None:
        return BoxResult(2, "RS/RW", "MANUAL", "RS data not available")

    if t.rs_slope_10d > 0:
        return BoxResult(
            2, "RS/RW", "PASS",
            f"RS slope +{t.rs_slope_10d:.2f}% over 10d (outperforming SPY)",
        )
    return BoxResult(
        2, "RS/RW", "FAIL",
        f"RS slope {t.rs_slope_10d:.2f}% over 10d (underperforming SPY)",
    )


def _box3_base_quality(t: TechnicalData) -> BoxResult:
    """BB squeeze, volume contracting, SMA stack intact."""
    reasons: list[str] = []
    fails: list[str] = []

    # SMA stack: 5 > 10 > 20 > 50, all rising
    smas = [t.sma_5, t.sma_10, t.sma_20, t.sma_50]
    if all(s is not None for s in smas):
        if smas[0] > smas[1] > smas[2] > smas[3]:  # type: ignore[operator]
            reasons.append("SMA stack intact (5>10>20>50)")
        else:
            fails.append("SMA stack broken")
    else:
        fails.append("SMA data insufficient for stack check")

    # Bollinger Band squeeze
    if t.bb_width is not None and t.bb_width_avg_20 is not None:
        if t.bb_width < t.bb_width_avg_20:
            reasons.append(f"BB squeeze ({t.bb_width:.4f} < avg {t.bb_width_avg_20:.4f})")
        else:
            fails.append(f"No BB squeeze ({t.bb_width:.4f} >= avg {t.bb_width_avg_20:.4f})")

    # Volume contraction (VDU)
    if t.volume_contracting is True:
        reasons.append("Volume contracting (VDU)")
    elif t.volume_contracting is False:
        fails.append("Volume not contracting")

    if fails:
        return BoxResult(3, "Base Quality", "FAIL", "; ".join(fails))
    if not reasons:
        return BoxResult(3, "Base Quality", "MANUAL", "Insufficient data")
    return BoxResult(3, "Base Quality", "PASS", "; ".join(reasons))


def _box4_catalyst() -> BoxResult:
    """Catalyst evaluation requires qualitative assessment — always MANUAL.
    Will be evaluated by Claude in E3-S2."""
    return BoxResult(4, "Catalyst", "MANUAL", "Requires qualitative assessment (news, earnings, sector theme)")


def _box5_risk(t: TechnicalData, price: float | None) -> BoxResult:
    """Stop distance ≤ 7% from entry, position size at 0.5% risk."""
    if price is None or price <= 0:
        return BoxResult(5, "Risk", "MANUAL", "Price not available")

    # Stop at 20 SMA (most conservative reference)
    stop_ref = t.sma_20
    if stop_ref is None:
        return BoxResult(5, "Risk", "MANUAL", "20 SMA not available for stop calculation")

    stop_distance_pct = (price - stop_ref) / price * 100

    if stop_distance_pct <= 0:
        return BoxResult(
            5, "Risk", "FAIL",
            f"Price {price:.2f} below 20 SMA {stop_ref:.2f} — no valid long stop",
        )

    if stop_distance_pct > MAX_STOP_PCT:
        return BoxResult(
            5, "Risk", "FAIL",
            f"Stop distance {stop_distance_pct:.1f}% exceeds {MAX_STOP_PCT}% limit (20 SMA at {stop_ref:.2f})",
        )

    return BoxResult(
        5, "Risk", "PASS",
        f"Stop at 20 SMA ({stop_ref:.2f}), distance {stop_distance_pct:.1f}%, within {MAX_STOP_PCT}% limit",
    )


def _sort_key(sc: ScoredCandidate) -> tuple[int, float]:
    """Sort by score descending, then RS strength descending."""
    rs = sc.enriched.technicals.rs_slope_10d if sc.enriched.technicals else 0
    return (sc.score, rs or 0)
