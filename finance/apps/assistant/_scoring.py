"""
finance.apps.assistant._scoring
================================
Weighted 0–100 scoring engine for swing trade candidates.

Each of the five dimensions produces a sub-score (0–weight) via piecewise
linear interpolation over anchored breakpoints. Hard gates zero a dimension
before sub-component averaging. Missing data is excluded from the average
and the remaining components are reweighted over available ones only.

Full spec: investing_framework/ScoringSystem.md
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

from finance.apps.analyst._models import Candidate, EnrichedCandidate, TechnicalData
from finance.apps.assistant._models import (
    CandidateScore,
    ComponentScore,
    DimensionScore,
    ScoringConfig,
)
from finance.apps.assistant._tags import _is_squeeze_on, _parse_squeeze_state

# ---------------------------------------------------------------------------
# _lerp — piecewise linear interpolation
# ---------------------------------------------------------------------------

# Anchors are (x, y) pairs. They must be sorted by x (ascending).
Anchors = Sequence[tuple[float, float]]


def _lerp(value: float, anchors: Anchors) -> float:
    """
    Piecewise linear interpolation over anchor points.

    Parameters
    ----------
    value:
        The input value to map.
    anchors:
        Sequence of (x, y) pairs sorted by x in ascending order.
        x values must be strictly monotonically increasing.
        y values may increase or decrease.

    Returns
    -------
    float
        Interpolated y in [0.0, 1.0], clamped at boundary anchors.
    """
    if not anchors:
        return 0.0

    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]

    if value <= xs[0]:
        return float(ys[0])
    if value >= xs[-1]:
        return float(ys[-1])

    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return float(ys[i] + t * (ys[i + 1] - ys[i]))

    return float(ys[-1])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _slope_category(slope_val: float | None, threshold: float) -> str | None:
    """Convert a numeric scanner slope value to a categorical slope string."""
    if slope_val is None:
        return None
    if abs(slope_val) <= threshold:
        return "flat"
    return "rising" if slope_val > 0 else "falling"


def _slope_score(slope: str | None, invert: bool = False) -> float | None:
    """
    Map a slope category string to a [0, 1] score.

    Parameters
    ----------
    invert:
        When True, flip the mapping (for short direction).
    """
    if slope is None:
        return None
    mapping = {"rising": 1.0, "flat": 0.5, "falling": 0.0}
    raw = mapping.get(slope.lower())
    if raw is None:
        return None
    if invert:
        raw = 1.0 - raw
    return raw


def _days_until_earnings(latest_earnings: str | None) -> int | None:
    """
    Return calendar days from today until the next earnings date.

    Accepts ISO (YYYY-MM-DD) and legacy (MM/DD/YY) formats.
    Returns None if parsing fails or the field is absent.
    Returns negative values when earnings are in the past.
    """
    if not latest_earnings:
        return None
    today = date.today()
    for fmt in ("%Y-%m-%d", "%m/%d/%y"):
        try:
            dt = datetime.strptime(latest_earnings, fmt).date()
            return (dt - today).days
        except ValueError:
            continue
    return None


def _make_component(
    name: str,
    raw_score: float | None,
    source: str,
) -> ComponentScore:
    """Build a ComponentScore, marking unavailable if raw_score is None."""
    if raw_score is None:
        return ComponentScore(name=name, raw_score=0.0, available=False, source="none")
    return ComponentScore(
        name=name,
        raw_score=_clamp01(raw_score),
        available=True,
        source=source,
    )


def _aggregate_components(
    components: list[ComponentScore],
    weight: int,
) -> tuple[float, float, bool]:
    """
    Compute (raw_score, weighted_score, partial) from a list of ComponentScores.

    Missing components (available=False) are excluded from the average.
    If ALL are unavailable, returns (0.0, 0.0, True) — all data absent, flagged partial.
    """
    available = [c for c in components if c.available]
    if not available:
        return 0.0, 0.0, True   # all unavailable — partial per DimensionScore spec
    partial = len(available) < len(components)
    raw = sum(c.raw_score for c in available) / len(available)
    weighted = raw * weight
    return raw, weighted, partial


# ---------------------------------------------------------------------------
# D1 — Trend Template (weight 25)
# ---------------------------------------------------------------------------

def _score_d1(
    c: Candidate,
    t: TechnicalData | None,
    direction: str,
    config: ScoringConfig,
) -> DimensionScore:
    """Trend Template: price trend, SMA alignment, 52W high/low, 12M return."""
    short = direction == "short"
    weight = config.weights[1]

    components: list[ComponentScore] = []

    # --- Price vs 50d SMA ---
    pct_vs_50 = _d1_price_vs_sma(c, t, short)
    components.append(_make_component("price_vs_50d_sma", pct_vs_50, "ibkr" if t else "scanner"))

    # --- 50d SMA slope ---
    slope_50 = _d1_slope_50(c, t, short, config.slope_flat_threshold)
    components.append(_make_component("slope_50d_sma", slope_50, "ibkr" if t and t.sma_50_slope else "scanner"))

    # --- Price vs 200d SMA ---
    pct_vs_200 = _d1_price_vs_200(c, t, short)
    components.append(_make_component("price_vs_200d_sma", pct_vs_200, "ibkr" if t else "scanner"))

    # --- 200d SMA slope ---
    slope_200 = _d1_slope_200(c, t, short, config.slope_flat_threshold)
    components.append(_make_component("slope_200d_sma", slope_200, "ibkr" if t and t.sma_200_slope else "scanner"))

    # --- 52W high/low distance ---
    high_low = _d1_52w_distance(c, t, short)
    components.append(_make_component("52w_distance", high_low, "ibkr" if t else "scanner"))

    # --- 12-month return ---
    ret_12m = _d1_12m_return(c, t, short)
    components.append(_make_component("return_12m", ret_12m, "ibkr" if t and t.return_12m is not None else "scanner"))

    raw, weighted, partial = _aggregate_components(components, weight)
    return DimensionScore(
        dimension=1,
        name="Trend Template",
        raw_score=raw,
        weighted_score=weighted,
        components=components,
        hard_gate_fired=False,
        partial=partial,
    )


def _d1_price_vs_sma(c: Candidate, t: TechnicalData | None, short: bool) -> float | None:
    """Price vs 50d SMA sub-component. Returns None when no data available."""
    if t is not None and t.sma_50 is not None and c.price is not None:
        pct = (c.price - t.sma_50) / t.sma_50 * 100
    elif c.pct_from_50d_sma is not None:
        pct = c.pct_from_50d_sma
    else:
        return None
    if not short:
        # Long: above SMA is good
        return _lerp(pct, [(-10, 0.0), (-2, 0.5), (0, 1.0)])
    else:
        # Short: below SMA is good
        return _lerp(pct, [(-10, 1.0), (-2, 0.5), (0, 0.5), (10, 0.0)])


def _d1_slope_50(
    c: Candidate, t: TechnicalData | None, short: bool, threshold: float
) -> float | None:
    if t is not None and t.sma_50_slope is not None:
        slope_str = t.sma_50_slope
    elif c.slope_50d_sma is not None:
        slope_str = _slope_category(c.slope_50d_sma, threshold)
    else:
        return None
    return _slope_score(slope_str, invert=short)


def _d1_price_vs_200(c: Candidate, t: TechnicalData | None, short: bool) -> float | None:
    if t is not None and t.sma_200 is not None and c.price is not None:
        pct = (c.price - t.sma_200) / t.sma_200 * 100
    else:
        # No direct scanner field for price vs 200d SMA; skip
        return None
    if not short:
        return _lerp(pct, [(-20, 0.0), (-5, 0.5), (0, 1.0)])
    else:
        return _lerp(pct, [(-20, 1.0), (-5, 0.5), (0, 0.5), (20, 0.0)])


def _d1_slope_200(
    c: Candidate, t: TechnicalData | None, short: bool, threshold: float
) -> float | None:
    if t is not None and t.sma_200_slope is not None:
        slope_str = t.sma_200_slope
    elif c.slope_200d_sma is not None:
        slope_str = _slope_category(c.slope_200d_sma, threshold)
    else:
        return None
    return _slope_score(slope_str, invert=short)


def _d1_52w_distance(c: Candidate, t: TechnicalData | None, short: bool) -> float | None:
    # Long: 52W high distance (how far below the 52W high)
    # Short: 52W low distance (how far above the 52W low)
    if not short:
        if t is not None and t.high_52w is not None and c.price is not None:
            dist = abs((c.price - t.high_52w) / t.high_52w * 100)
        elif c.high_52w_distance_pct is not None:
            dist = abs(c.high_52w_distance_pct)  # stored as negative; abs gives distance
        else:
            return None
        return _lerp(dist, _52W_DISTANCE_ANCHORS)
    else:
        # Short: proximity to 52W low is the equivalent signal
        if t is not None and t.low_52w is not None and c.price is not None:
            dist = abs((c.price - t.low_52w) / t.low_52w * 100)
        else:
            return None  # no scanner fallback for 52W low distance
        return _lerp(dist, _52W_DISTANCE_ANCHORS)


def _d1_12m_return(c: Candidate, t: TechnicalData | None, short: bool) -> float | None:
    if t is not None and t.return_12m is not None:
        ret = t.return_12m
    elif c.weighted_alpha is not None:
        ret = c.weighted_alpha  # proxy: Wtd Alpha is momentum-weighted 12M return
    else:
        return None
    if not short:
        return _lerp(ret, [(-20, 0.0), (0, 0.5), (20, 1.0)])
    else:
        return _lerp(ret, [(-20, 1.0), (0, 0.5), (20, 0.0)])


# ---------------------------------------------------------------------------
# D2 — Relative Strength (weight 25)
# ---------------------------------------------------------------------------

def _score_d2(
    c: Candidate,
    t: TechnicalData | None,
    direction: str,
    config: ScoringConfig,
) -> DimensionScore:
    """Relative Strength: RS slope vs SPY (IBKR) + perf vs market (scanner)."""
    short = direction == "short"
    weight = config.weights[2]
    components: list[ComponentScore] = []

    # --- RS slope vs SPY (IBKR only) ---
    rs_slope_score: float | None = None
    if t is not None and t.rs_slope_10d is not None:
        raw = _lerp(t.rs_slope_10d, [(-0.5, 0.0), (0.0, 0.3), (0.5, 1.0)])
        rs_slope_score = (1.0 - raw) if short else raw
    components.append(_make_component("rs_slope_10d", rs_slope_score, "ibkr"))

    # --- Perf vs market 5D ---
    p5 = _perf_component(c.perf_vs_market_5d, [(-5, 0.0), (0, 0.5), (5, 1.0)], short)
    components.append(_make_component("perf_vs_market_5d", p5, "scanner" if p5 is not None else "none"))

    # --- Perf vs market 1M ---
    p1m = _perf_component(c.perf_vs_market_1m, [(-5, 0.0), (0, 0.5), (5, 1.0)], short)
    components.append(_make_component("perf_vs_market_1m", p1m, "scanner" if p1m is not None else "none"))

    # --- Perf vs market 3M ---
    p3m = _perf_component(c.perf_vs_market_3m, [(-10, 0.0), (0, 0.5), (10, 1.0)], short)
    components.append(_make_component("perf_vs_market_3m", p3m, "scanner" if p3m is not None else "none"))

    raw, weighted, partial = _aggregate_components(components, weight)
    return DimensionScore(
        dimension=2,
        name="Relative Strength",
        raw_score=raw,
        weighted_score=weighted,
        components=components,
        hard_gate_fired=False,
        partial=partial,
    )


def _perf_component(
    value: float | None, anchors: Anchors, short: bool
) -> float | None:
    if value is None:
        return None
    raw = _lerp(value, anchors)
    return (1.0 - raw) if short else raw


# ---------------------------------------------------------------------------
# D3 — Base Quality (weight 15)
# ---------------------------------------------------------------------------

def _score_d3(
    c: Candidate,
    t: TechnicalData | None,
    direction: str,
    config: ScoringConfig,
) -> DimensionScore:
    """Base Quality: BB squeeze, volume, SMA stack, ADR%."""
    short = direction == "short"
    weight = config.weights[3]
    components: list[ComponentScore] = []

    # --- BB squeeze ---
    bb_score, bb_source = _d3_bb_squeeze(c, t, short)
    components.append(_make_component("bb_squeeze", bb_score, bb_source))

    # --- Volume (VDU / RVOL) ---
    vol_score, vol_source = _d3_volume(c, t, short)
    components.append(_make_component("volume_vdu", vol_score, vol_source))

    # --- SMA stack ---
    stack_score, stack_source = _d3_sma_stack(c, t, short)
    components.append(_make_component("sma_stack", stack_score, stack_source))

    # --- ADR% ---
    adr_score = _d3_adr(c)
    components.append(_make_component("adr_pct", adr_score, "scanner" if adr_score is not None else "none"))

    raw, weighted, partial = _aggregate_components(components, weight)
    return DimensionScore(
        dimension=3,
        name="Base Quality",
        raw_score=raw,
        weighted_score=weighted,
        components=components,
        hard_gate_fired=False,
        partial=partial,
    )


def _d3_bb_squeeze(
    c: Candidate, t: TechnicalData | None, short: bool
) -> tuple[float | None, str]:
    """
    BB squeeze sub-component.

    IBKR path (bb_width available):
      Long:  squeeze_on→1.0, expanding→0.8, else→0.3
      Short: expanding→0.8, squeeze_on→0.5, else→0.3

    Scanner path (4-state ttm_squeeze):
      Long:
        fired_long  → 1.0
        fired_short → 0.5 (fired wrong direction)
        on          → 0.8 (building, not yet fired)
        off+wide    → 0.8 (recently fired proxy)
        off+narrow  → 0.3
      Short:
        fired_short → 1.0
        fired_long  → 0.5 (fired wrong direction)
        on          → 0.5 (contraction, potential reversal)
        off+wide    → 0.8 (distribution expanding)
        off+narrow  → 0.3
    """
    if t is not None and t.bb_width is not None and t.bb_width_avg_20 is not None:
        squeeze_on = t.bb_width < t.bb_width_avg_20
        bb_expanding = t.bb_width > t.bb_width_avg_20
        source = "ibkr"
        if not short:
            if squeeze_on:
                return 1.0, source
            if bb_expanding:
                return 0.8, source
            return 0.3, source
        else:
            if bb_expanding:
                return 0.8, source
            if squeeze_on:
                return 0.5, source
            return 0.3, source

    # Scanner fallback — 4-state
    state = _parse_squeeze_state(c.ttm_squeeze)
    if state is None:
        return None, "none"

    bb_wide = c.bb_pct is not None and c.bb_pct > 80
    source = "scanner"

    if not short:
        if state == "fired_long":
            return 1.0, source
        if state == "fired_short":
            return 0.5, source
        if state == "on":
            return 0.8, source
        # off
        return (0.8 if bb_wide else 0.3), source
    else:
        if state == "fired_short":
            return 1.0, source
        if state == "fired_long":
            return 0.5, source
        if state == "on":
            return 0.5, source
        # off
        return (0.8 if bb_wide else 0.3), source


def _d3_volume(
    c: Candidate, t: TechnicalData | None, short: bool
) -> tuple[float | None, str]:
    """Volume sub-component (VDU from IBKR, or RVOL from scanner)."""
    if t is not None and t.volume_contracting is not None:
        # IBKR VDU flag takes priority
        if not short:
            if t.volume_contracting:
                return 1.0, "ibkr"
            # Not contracting — use RVOL for banded scoring (spec: RVOL <0.8→0.8, 0.8–1.2→0.5, >1.5→0.2)
            rvol = t.rvol
            if rvol is not None:
                if rvol < 0.8:
                    return 0.8, "ibkr"
                if rvol <= 1.2:
                    return 0.5, "ibkr"
                if rvol > 1.5:
                    return 0.2, "ibkr"
                return 0.4, "ibkr"
            return 0.2, "ibkr"
        else:
            # Short: expanding volume (distribution) is good
            rvol = t.rvol
            if rvol is not None:
                if rvol > 1.5:
                    return 0.8, "ibkr"
                if 0.8 <= rvol <= 1.5:
                    return 0.5, "ibkr"
                return 0.3, "ibkr"
            return (0.3 if t.volume_contracting else 0.8), "ibkr"

    if c.rvol_20d is None:
        return None, "none"

    rvol = c.rvol_20d
    source = "scanner"
    if not short:
        # Long: low RVOL = VDU-like supply exhaustion
        if rvol < 0.8:
            return 0.8, source
        if rvol <= 1.2:
            return 0.5, source
        if rvol > 1.5:
            return 0.2, source
        return 0.4, source
    else:
        # Short: high RVOL = institutional distribution
        if rvol > 1.5:
            return 0.8, source
        if 0.8 <= rvol <= 1.5:
            return 0.5, source
        return 0.3, source


def _d3_sma_stack(
    c: Candidate, t: TechnicalData | None, short: bool
) -> tuple[float | None, str]:
    """SMA stack alignment sub-component (IBKR only for full count)."""
    if t is None:
        return None, "none"
    if c.price is None:
        return None, "none"

    smas = [t.sma_5, t.sma_10, t.sma_20, t.sma_50]
    if any(s is None for s in smas):
        return None, "none"

    sma5, sma10, sma20, sma50 = smas  # type: ignore[misc]
    price = c.price

    if not short:
        # Long: price > sma5 > sma10 > sma20 > sma50
        aligned = [
            price > sma5,
            sma5 > sma10,
            sma10 > sma20,
            sma20 > sma50,
        ]
    else:
        # Short: price < sma5 < sma10 < sma20 < sma50
        aligned = [
            price < sma5,
            sma5 < sma10,
            sma10 < sma20,
            sma20 < sma50,
        ]

    count = sum(aligned)
    score = _lerp(count, [(0, 0.0), (2, 0.3), (3, 0.6), (4, 1.0)])
    return score, "ibkr"


def _d3_adr(c: Candidate) -> float | None:
    """
    ADR% sub-component. Same for long and short.
    3–7% → 1.0; 2–3% or 7–10% → 0.6; <2% or >10% → 0.2
    """
    adr = c.adr_pct_20d
    if adr is None:
        return None
    if 3.0 <= adr <= 7.0:
        return 1.0
    if 2.0 <= adr < 3.0 or 7.0 < adr <= 10.0:
        return 0.6
    return 0.2


# ---------------------------------------------------------------------------
# D4 — Catalyst (weight 20)
# ---------------------------------------------------------------------------

_52W_DISTANCE_ANCHORS: list[tuple[float, float]] = [
    (0, 1.0), (5, 0.8), (15, 0.4), (25, 0.1), (30, 0.0)
]

_BLACKOUT_DAYS_LONG = 5
_BLACKOUT_DAYS_SHORT = 10


def _score_d4(
    c: Candidate,
    direction: str,
    config: ScoringConfig,
) -> DimensionScore:
    """Catalyst: earnings proximity (hard gate), surprise, history, P/C, RVOL."""
    short = direction == "short"
    weight = config.weights[4]
    blackout = _BLACKOUT_DAYS_SHORT if short else _BLACKOUT_DAYS_LONG

    # Compute all sub-components first (for display), then check gate
    days = _days_until_earnings(c.latest_earnings)
    components: list[ComponentScore] = []

    # --- Earnings proximity ---
    prox_score: float | None
    if days is None:
        prox_score = None
    else:
        if days < blackout:
            prox_score = None  # hard gate; mark as available=False to surface gate reason
        elif not short:
            prox_score = _lerp(days, [(5, 0.5), (10, 0.8), (20, 1.0)])
        else:
            prox_score = _lerp(days, [(10, 0.5), (20, 1.0)])
    components.append(_make_component("earnings_proximity", prox_score, "scanner"))

    # Hard gate evaluation
    hard_gate = days is not None and days < blackout
    if hard_gate:
        # Still compute remaining components for display, then zero the dimension
        components.extend(_d4_remaining_components(c, short))
        return DimensionScore(
            dimension=4,
            name="Catalyst",
            raw_score=0.0,
            weighted_score=0.0,
            components=components,
            hard_gate_fired=True,
            partial=False,
        )

    components.extend(_d4_remaining_components(c, short))
    raw, weighted, partial = _aggregate_components(components, weight)
    return DimensionScore(
        dimension=4,
        name="Catalyst",
        raw_score=raw,
        weighted_score=weighted,
        components=components,
        hard_gate_fired=False,
        partial=partial,
    )


def _d4_remaining_components(c: Candidate, short: bool) -> list[ComponentScore]:
    """Earnings surprise, history, P/C ratio, and RVOL components."""
    comps: list[ComponentScore] = []

    # --- Earnings surprise ---
    surprise = _d4_surprise(c, short)
    comps.append(_make_component("earnings_surprise", surprise, "scanner" if c.earnings_surprise_pct is not None else "none"))

    # --- Surprise history (4-quarter beat/miss record) ---
    history = _d4_surprise_history(c, short)
    all_hist = all(v is not None for v in [c.earnings_surprise_pct, c.earnings_surprise_q1, c.earnings_surprise_q2, c.earnings_surprise_q3])
    comps.append(_make_component("surprise_history", history, "scanner" if all_hist else "none"))

    # --- Put/Call ratio ---
    pc_score = _d4_put_call(c, short)
    comps.append(_make_component("put_call_ratio", pc_score, "scanner" if c.put_call_vol_5d is not None else "none"))

    # --- RVOL ---
    rvol_score = _d4_rvol(c, short)
    comps.append(_make_component("rvol", rvol_score, "scanner" if c.rvol_20d is not None else "none"))

    return comps


def _d4_surprise(c: Candidate, short: bool) -> float | None:
    s = c.earnings_surprise_pct
    if s is None:
        return None
    if not short:
        # Long: beat = good
        if s >= 10.0:
            return 1.0
        if s >= 5.0:
            return _lerp(s, [(5, 0.7), (10, 1.0)])
        if s >= 0.0:
            return _lerp(s, [(0, 0.3), (5, 0.7)])
        return 0.0
    else:
        # Short: miss = good
        if s <= -10.0:
            return 1.0
        if s <= -5.0:
            return _lerp(s, [(-10, 1.0), (-5, 0.7)])
        if s <= 0.0:
            return _lerp(s, [(-5, 0.7), (0, 0.3)])
        return 0.0


def _d4_surprise_history(c: Candidate, short: bool) -> float | None:
    """Count beats or misses across 4 quarters."""
    quarters = [
        c.earnings_surprise_pct,
        c.earnings_surprise_q1,
        c.earnings_surprise_q2,
        c.earnings_surprise_q3,
    ]
    available = [q for q in quarters if q is not None]
    if not available:
        return None

    if not short:
        beats = sum(1 for q in available if q > 0)
        total = len(available)
        # Scale to 4-quarter equivalent
        ratio = beats / total
        if ratio == 1.0:
            return 1.0
        if ratio >= 0.75:
            return 0.8
        if ratio >= 0.5:
            return 0.5
        return 0.2
    else:
        # Short: count misses
        curr = c.earnings_surprise_pct
        prior = [c.earnings_surprise_q1, c.earnings_surprise_q2, c.earnings_surprise_q3]
        prior_available = [q for q in prior if q is not None]

        if curr is None:
            return None

        curr_miss = curr < 0
        prior_misses = sum(1 for q in prior_available if q < 0)

        if curr_miss and prior_misses == 0:
            return 1.0   # first miss — strongest drift
        if curr_miss and prior_misses >= 1:
            return 0.7   # consecutive miss
        return 0.0       # beat


def _d4_put_call(c: Candidate, short: bool) -> float | None:
    pc = c.put_call_vol_5d
    if pc is None:
        return None
    if not short:
        return _lerp(pc, [(0.0, 1.0), (0.3, 1.0), (0.5, 0.8), (1.0, 0.3), (1.5, 0.2), (2.0, 0.0)])
    else:
        return _lerp(pc, [(0.0, 0.0), (0.5, 0.2), (1.0, 0.3), (1.5, 0.8), (2.0, 1.0)])


def _d4_rvol(c: Candidate, short: bool) -> float | None:
    rvol = c.rvol_20d
    if rvol is None:
        return None
    if not short:
        return _lerp(rvol, [(0.8, 0.0), (1.0, 0.3), (1.5, 0.5), (2.0, 0.7), (3.0, 1.0)])
    else:
        return _lerp(rvol, [(0.8, 0.0), (1.0, 0.3), (1.5, 0.7), (2.0, 1.0)])


# ---------------------------------------------------------------------------
# D5 — Risk (weight 15)
# ---------------------------------------------------------------------------

_STOP_GATE_PCT = 7.0
_SHORT_FLOAT_GATE_PCT = 20.0


def _score_d5(
    c: Candidate,
    t: TechnicalData | None,
    direction: str,
    config: ScoringConfig,
) -> DimensionScore:
    """Risk: stop distance (hard gate), ADR vs stop, market cap, short float (shorts)."""
    short = direction == "short"
    weight = config.weights[5]
    components: list[ComponentScore] = []

    # --- Stop distance ---
    raw_stop_pct = _d5_raw_stop_pct(c, t)
    stop_gate = raw_stop_pct is not None and raw_stop_pct > _STOP_GATE_PCT
    stop_dist = _d5_stop_distance(c, t)
    components.append(_make_component("stop_distance", stop_dist, "ibkr" if t and t.sma_20 else "scanner"))

    # --- ADR vs stop ---
    adr_stop_score = _d5_adr_vs_stop(c, t)
    components.append(_make_component("adr_vs_stop", adr_stop_score, "scanner"))

    # --- Market cap ---
    cap_score = _d5_market_cap(c)
    components.append(_make_component("market_cap", cap_score, "scanner" if c.market_cap_k is not None else "none"))

    # --- Short float (shorts only) ---
    short_float_gate = False
    if short:
        sf_score = _d5_short_float(c)
        components.append(_make_component("short_float", sf_score, "scanner" if c.short_float is not None else "none"))
        if c.short_float is not None and c.short_float > _SHORT_FLOAT_GATE_PCT:
            short_float_gate = True

    hard_gate = stop_gate or short_float_gate
    if hard_gate:
        return DimensionScore(
            dimension=5,
            name="Risk",
            raw_score=0.0,
            weighted_score=0.0,
            components=components,
            hard_gate_fired=True,
            partial=False,
        )

    raw, weighted, partial = _aggregate_components(components, weight)
    return DimensionScore(
        dimension=5,
        name="Risk",
        raw_score=raw,
        weighted_score=weighted,
        components=components,
        hard_gate_fired=False,
        partial=partial,
    )


def _d5_raw_stop_pct(c: Candidate, t: TechnicalData | None) -> float | None:
    """Return the raw stop distance percentage (not scored)."""
    if t is not None and t.sma_20 is not None and c.price is not None and c.price > 0:
        return abs((c.price - t.sma_20) / c.price * 100)
    if c.atr_pct_20d is not None:
        return c.atr_pct_20d
    return None


def _d5_stop_distance(c: Candidate, t: TechnicalData | None) -> float | None:
    pct = _d5_raw_stop_pct(c, t)
    if pct is None:
        return None
    if pct > _STOP_GATE_PCT:
        return 0.0  # gate signal — will be caught in _score_d5
    return _lerp(pct, [(0, 1.0), (2, 1.0), (3, 0.8), (5, 0.5), (7, 0.1)])


def _d5_adr_vs_stop(c: Candidate, t: TechnicalData | None) -> float | None:
    """ADR vs stop distance ratio. Lower ratio (tighter stop relative to ADR) = better."""
    adr = c.adr_pct_20d
    if adr is None or adr == 0:
        return None
    stop_pct = _d5_raw_stop_pct(c, t)
    if stop_pct is None:
        return None
    ratio = stop_pct / adr
    return _lerp(ratio, [(0, 1.0), (0.5, 1.0), (1.0, 0.7), (1.5, 0.4), (2.0, 0.1)])


def _d5_market_cap(c: Candidate) -> float | None:
    cap = c.market_cap_k  # value in dollars (despite field name)
    if cap is None:
        return None
    BILLION = 1_000_000_000
    return _lerp(cap, [
        (200_000_000, 0.3),
        (500_000_000, 0.5),
        (2 * BILLION, 0.8),
        (10 * BILLION, 1.0),
    ])


def _d5_short_float(c: Candidate) -> float | None:
    sf = c.short_float
    if sf is None:
        return None
    if sf > _SHORT_FLOAT_GATE_PCT:
        return 0.0  # gate signal — caught in _score_d5
    return _lerp(sf, [(0, 1.0), (5, 1.0), (10, 0.7), (15, 0.4), (20, 0.1)])


# ---------------------------------------------------------------------------
# score_candidate — main entry point
# ---------------------------------------------------------------------------

def score_candidate(
    enriched: EnrichedCandidate,
    direction: str,
    tags: list[str],
    config: ScoringConfig,
) -> CandidateScore:
    """
    Score a single enriched candidate on all five dimensions.

    Parameters
    ----------
    enriched:
        Candidate with optional IBKR technical data.
    direction:
        "long" or "short" — derived from tag assignment before calling.
    tags:
        List of scanner tags already assigned to this candidate.
    config:
        Scoring weights and bonus configuration.

    Returns
    -------
    CandidateScore
        Dimension scores, tag bonus, and total.
    """
    c = enriched.candidate
    t = enriched.technicals if enriched.data_available else None

    d1 = _score_d1(c, t, direction, config)
    d2 = _score_d2(c, t, direction, config)
    d3 = _score_d3(c, t, direction, config)
    d4 = _score_d4(c, direction, config)
    d5 = _score_d5(c, t, direction, config)

    dimensions = [d1, d2, d3, d4, d5]
    dim_total = sum(d.weighted_score for d in dimensions)

    tag_bonus = min(len(tags) * config.tag_bonus_per_tag, config.tag_bonus_cap)
    total = dim_total + tag_bonus

    return CandidateScore(
        direction=direction,
        dimensions=dimensions,
        tag_bonus=float(tag_bonus),
        total=total,
        tags=list(tags),
    )
