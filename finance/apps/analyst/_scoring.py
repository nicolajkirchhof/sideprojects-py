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
from datetime import date, datetime

from finance.apps.analyst._models import (
    BoxResult,
    Candidate,
    EnrichedCandidate,
    ScoredCandidate,
    TechnicalData,
)

log = logging.getLogger(__name__)

MAX_STOP_PCT = 7.0       # Minervini hard limit: stop ≤ 7% from entry
MAX_52W_DISTANCE = 25.0  # Price within 25% of 52W high
EARNINGS_BLACKOUT_DAYS = 5  # No entries within 5 days of earnings


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

    c = ec.candidate
    boxes = [
        _box1_trend_template(t, c),
        _box2_relative_strength(t),
        _box3_base_quality(t, c),
        _box4_catalyst(c),
        _box5_risk(t, c),
    ]
    passed = sum(1 for b in boxes if b.status == "PASS")
    return ScoredCandidate(enriched=ec, boxes=boxes, score=passed)


def _box1_trend_template(t: TechnicalData, c: Candidate) -> BoxResult:
    """Minervini Trend Template: Price > 20 SMA > 50 SMA, 50 SMA rising,
    within 25% of 52W high, positive 12-month return."""
    p = c.price
    if p is None or p <= 0:
        return BoxResult(1, "Trend Template", "MANUAL", "Price not available")

    reasons: list[str] = []
    fails: list[str] = []

    # SMA stack check (IBKR data, fallback to scanner % from 50D SMA)
    if t.sma_20 is not None and t.sma_50 is not None:
        if p > t.sma_20 > t.sma_50:
            reasons.append(f"Price {p:.2f} > 20 SMA {t.sma_20:.2f} > 50 SMA {t.sma_50:.2f}")
        else:
            fails.append(f"SMA stack broken: Price {p:.2f}, 20 SMA {t.sma_20:.2f}, 50 SMA {t.sma_50:.2f}")
    elif c.pct_from_50d_sma is not None:
        if c.pct_from_50d_sma > 0:
            reasons.append(f"Price {c.pct_from_50d_sma:+.1f}% above 50D SMA (scanner)")
        else:
            fails.append(f"Price {c.pct_from_50d_sma:.1f}% below 50D SMA (scanner)")
    else:
        fails.append("SMA data insufficient")

    if t.sma_50_slope == "rising":
        reasons.append("50 SMA rising")
    elif t.sma_50_slope is not None:
        fails.append(f"50 SMA {t.sma_50_slope}")

    # 52W high distance (IBKR, fallback to scanner)
    if t.high_52w is not None and p > 0:
        distance = (t.high_52w - p) / t.high_52w * 100
        if distance <= MAX_52W_DISTANCE:
            reasons.append(f"Within {distance:.1f}% of 52W high")
        else:
            fails.append(f"{distance:.1f}% below 52W high (max {MAX_52W_DISTANCE}%)")
    elif c.high_52w_distance_pct is not None:
        dist = abs(c.high_52w_distance_pct)
        if dist <= MAX_52W_DISTANCE:
            reasons.append(f"Within {dist:.1f}% of 52W high (scanner)")
        else:
            fails.append(f"{dist:.1f}% below 52W high (max {MAX_52W_DISTANCE}%, scanner)")

    # 12-month return / century momentum (IBKR, fallback to scanner 52W %Chg)
    return_12m = t.return_12m if t.return_12m is not None else c.change_52w_pct
    if return_12m is not None:
        if return_12m > 0:
            reasons.append(f"12M return +{return_12m:.1f}%")
        else:
            fails.append(f"12M return {return_12m:.1f}% (negative)")

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


def _box3_base_quality(t: TechnicalData, c: Candidate) -> BoxResult:
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

    # Bollinger Band squeeze (IBKR, fallback to scanner BB%)
    if t.bb_width is not None and t.bb_width_avg_20 is not None:
        if t.bb_width < t.bb_width_avg_20:
            reasons.append(f"BB squeeze ({t.bb_width:.4f} < avg {t.bb_width_avg_20:.4f})")
        else:
            fails.append(f"No BB squeeze ({t.bb_width:.4f} >= avg {t.bb_width_avg_20:.4f})")
    elif c.bb_pct is not None:
        # BB% near 50 = middle of band, >100 = above upper, <0 = below lower
        reasons.append(f"BB% {c.bb_pct:.0f}% (scanner)")

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


def _box4_catalyst(c: Candidate) -> BoxResult:
    """Catalyst evaluation — auto-fail on imminent earnings, otherwise MANUAL with hints."""
    # Earnings blackout: auto-fail if earnings within 5 trading days
    if c.latest_earnings:
        earnings_days = _days_until_earnings(c.latest_earnings)
        if earnings_days is not None and 0 <= earnings_days <= EARNINGS_BLACKOUT_DAYS:
            return BoxResult(4, "Catalyst", "FAIL",
                f"Earnings in {earnings_days}d ({c.latest_earnings}) — no entries within {EARNINGS_BLACKOUT_DAYS}d")

    hints: list[str] = []

    # Options flow signals (PM-04)
    if c.put_call_vol_5d is not None:
        if c.put_call_vol_5d < 0.5:
            hints.append(f"Call-dominant flow (P/C {c.put_call_vol_5d:.2f})")
        elif c.put_call_vol_5d > 1.5:
            hints.append(f"Put-heavy flow (P/C {c.put_call_vol_5d:.2f}) — potential squeeze")

    # Earnings date (outside blackout — note for awareness)
    if c.latest_earnings:
        hints.append(f"Next earnings: {c.latest_earnings}")

    # Short squeeze potential
    if c.days_to_cover is not None and c.days_to_cover > 5:
        hints.append(f"Days to cover: {c.days_to_cover:.1f} (squeeze potential)")

    if hints:
        return BoxResult(4, "Catalyst", "MANUAL", "; ".join(hints) + " — needs qualitative review")
    return BoxResult(4, "Catalyst", "MANUAL", "No signals — needs qualitative review (news, earnings, sector theme)")


def _days_until_earnings(earnings_str: str) -> int | None:
    """Parse earnings date string (MM/DD/YY or MM/DD/YYYY) and return days until."""
    today = date.today()
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%m/%d/%y %H:%M"):
        try:
            earnings_date = datetime.strptime(earnings_str.strip(), fmt).date()
            return (earnings_date - today).days
        except ValueError:
            continue
    return None


def _box5_risk(t: TechnicalData, c: Candidate) -> BoxResult:
    """Stop distance ≤ 7% from entry, position size at 0.5% risk."""
    price = c.price
    if price is None or price <= 0:
        return BoxResult(5, "Risk", "MANUAL", "Price not available")

    # Stop at 20 SMA (most conservative reference)
    stop_ref = t.sma_20
    if stop_ref is None:
        # Fallback: use ATR% from scanner for rough stop estimate
        if c.atr_pct_20d is not None:
            if c.atr_pct_20d <= MAX_STOP_PCT:
                return BoxResult(5, "Risk", "PASS",
                    f"ATR {c.atr_pct_20d:.1f}% within {MAX_STOP_PCT}% limit (scanner, no SMA data)")
            return BoxResult(5, "Risk", "FAIL",
                f"ATR {c.atr_pct_20d:.1f}% exceeds {MAX_STOP_PCT}% limit (scanner, no SMA data)")
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
