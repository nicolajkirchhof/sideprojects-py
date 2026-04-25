"""
finance.apps.assistant._tags
=============================
Scanner tag assignment and direction derivation.

Tags are assigned from:
  1. Condition-based rules on Candidate fields (52w-high, 5d-momentum, etc.)
  2. Scanner membership sets (ep-gap, rw-breakdown, short-squeeze, high-put-ratio, high-call-ratio)

Full spec: finance/BACKLOG-ASSISTANT.md § TA-E1-S7
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from finance.apps.analyst._models import Candidate

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

SqueezeState = Literal["on", "fired_long", "fired_short", "off"]


def _parse_squeeze_state(ttm_squeeze) -> SqueezeState | None:
    """
    Parse Barchart TTM Squeeze field into a typed 4-state enum.

    Barchart exports:
      "On"    — squeeze building (consolidation underway)
      "N/A"   — squeeze off (no consolidation)
      "Long"  — squeeze fired long today (breakout signal)
      "Short" — squeeze fired short today (breakdown signal)

    Also handles legacy CSV numeric ("0"/"1") and boolean-like values.
    Returns None when the field is absent or unrecognised.
    """
    if ttm_squeeze is None:
        return None
    try:
        if pd.isna(ttm_squeeze):
            return None
    except (TypeError, ValueError):
        pass
    norm = str(ttm_squeeze).strip().lower()
    if norm in ("on", "1", "true"):
        return "on"
    if norm == "long":
        return "fired_long"
    if norm == "short":
        return "fired_short"
    if norm in ("off", "0", "false", "n/a", "na"):
        return "off"
    return None


def _is_squeeze_on(ttm_squeeze) -> bool | None:
    """
    Thin wrapper over _parse_squeeze_state() for callers that only need bool.

    Returns:
      True  — squeeze is active: "on", "fired_long", or "fired_short"
      False — squeeze is off: "off"
      None  — unknown / missing
    """
    state = _parse_squeeze_state(ttm_squeeze)
    if state is None:
        return None
    return state != "off"


# ---------------------------------------------------------------------------
# Scanner key → tag name mapping for membership-based tags
# ---------------------------------------------------------------------------

_MEMBERSHIP_TAG_MAP: dict[str, str] = {
    "ep-gap-scanner": "ep-gap",
    "rw-breakdown-candidates": "rw-breakdown",
    "short-squeeze": "short-squeeze",
    "high-put-ratio": "high-put-ratio",
    "high-call-ratio": "high-call-ratio",
}

# ---------------------------------------------------------------------------
# Tag rules
# ---------------------------------------------------------------------------

def _tag_52w_high(c: Candidate) -> bool:
    """Near 52W high with squeeze and elevated volume."""
    return (
        c.high_52w_distance_pct is not None
        and c.high_52w_distance_pct > -5.0    # within 5% of 52W high
        and _is_squeeze_on(c.ttm_squeeze)
        and c.rvol_20d is not None
        and c.rvol_20d > 1.0
    )


def _tag_5d_momentum(c: Candidate) -> bool:
    """Strong short-term momentum with market outperformance."""
    return (
        c.change_5d_pct is not None
        and c.change_5d_pct > 5.0
        and c.rvol_20d is not None
        and c.rvol_20d > 1.0
        and c.perf_vs_market_5d is not None
        and c.perf_vs_market_5d > 0.0
    )


def _tag_1m_strength(c: Candidate) -> bool:
    """One-month momentum with market outperformance and squeeze."""
    return (
        c.change_1m_pct is not None
        and c.change_1m_pct > 10.0
        and c.perf_vs_market_1m is not None
        and c.perf_vs_market_1m > 0.0
        and _is_squeeze_on(c.ttm_squeeze)
    )


def _tag_vol_spike(c: Candidate) -> bool:
    """Elevated relative volume indicating institutional participation."""
    return c.rvol_20d is not None and c.rvol_20d > 1.75


def _tag_trend_seeker(c: Candidate) -> bool:
    """Barchart Trend Seeker proprietary Buy signal."""
    return (
        c.trend_seeker_signal is not None
        and c.trend_seeker_signal.strip().lower() == "buy"
    )


def _bb_expanded(c: Candidate) -> bool:
    """True when BBands are expanded — prefers bb_pct, falls back to bb_rank."""
    if c.bb_pct is not None:
        return c.bb_pct > 80.0
    if c.bb_rank is not None:
        return c.bb_rank > 80.0
    return False


def _tag_ttm_fired(c: Candidate) -> bool:
    """
    TTM Squeeze fired signal.

    Triggered by:
      1. Barchart "Long" or "Short" state — squeeze explicitly fired today.
      2. Proxy: squeeze "Off" + BB expanded + RVOL > 1.0 + ATR% < 7%
         (recently-fired approximation when only "Off" is reported).

    BB expansion uses bb_pct when available, falling back to bb_rank (0–100).
    """
    state = _parse_squeeze_state(c.ttm_squeeze)
    if state is None:
        return False
    if state in ("fired_long", "fired_short", "off"):
        return (
            _bb_expanded(c)
            and c.rvol_20d is not None and c.rvol_20d > 1.0
            and c.atr_pct_20d is not None and c.atr_pct_20d < 7.0
        )
    return False


def _tag_pead_long(c: Candidate) -> bool:
    """Post-earnings announcement drift — bullish."""
    return (
        c.earnings_surprise_pct is not None
        and c.earnings_surprise_pct > 5.0
        and c.change_5d_pct is not None
        and c.change_5d_pct > 10.0
        and c.perf_vs_market_5d is not None
        and c.perf_vs_market_5d > 0.0
        and c.weighted_alpha is not None
        and c.weighted_alpha > 0.0
    )


def _tag_pead_short(c: Candidate) -> bool:
    """Post-earnings announcement drift — bearish (miss + high short float guard)."""
    return (
        c.earnings_surprise_pct is not None
        and c.earnings_surprise_pct < -5.0
        and c.change_5d_pct is not None
        and c.change_5d_pct < -5.0
        and c.pct_from_50d_sma is not None
        and c.pct_from_50d_sma < 0.0          # below 50d SMA
        and (c.short_float is None or c.short_float < 20.0)   # squeeze risk guard
    )


def _tag_consecutive_miss(c: Candidate) -> bool:
    """Consecutive earnings misses — persistent negative PEAD drift candidate."""
    if c.earnings_surprise_pct is None or c.earnings_surprise_pct >= 0:
        return False
    prior = [c.earnings_surprise_q1, c.earnings_surprise_q2, c.earnings_surprise_q3]
    prior_misses = sum(1 for q in prior if q is not None and q < 0)
    return prior_misses >= 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_tags(
    candidate: Candidate,
    scanner_sets: dict[str, set[str]],
) -> list[str]:
    """
    Assign scanner tags to a candidate.

    Parameters
    ----------
    candidate:
        Parsed Candidate with all available fields filled in.
    scanner_sets:
        Mapping of scanner key → set of symbols present in that scanner file.
        Used for membership-based tags (ep-gap, rw-breakdown, etc.).
        e.g. {"ep-gap-scanner": {"AAPL", "TSLA"}, ...}

    Returns
    -------
    list[str]
        Sorted list of assigned tag names.
    """
    tags: list[str] = []
    c = candidate

    # Condition-based tags (from Long Universe / PEAD scanner fields)
    if _tag_52w_high(c):
        tags.append("52w-high")
    if _tag_5d_momentum(c):
        tags.append("5d-momentum")
    if _tag_1m_strength(c):
        tags.append("1m-strength")
    if _tag_vol_spike(c):
        tags.append("vol-spike")
    if _tag_trend_seeker(c):
        tags.append("trend-seeker")
    if _tag_ttm_fired(c):
        tags.append("ttm-fired")
    if _tag_pead_long(c):
        tags.append("pead-long")
    if _tag_pead_short(c):
        tags.append("pead-short")
    if _tag_consecutive_miss(c):
        tags.append("consecutive-miss")

    # Membership-based tags (from scanner file participation)
    symbol = c.symbol.upper()
    for scanner_key, tag_name in _MEMBERSHIP_TAG_MAP.items():
        if symbol in scanner_sets.get(scanner_key, set()):
            tags.append(tag_name)

    return sorted(set(tags))  # deduplicated and stable-sorted


def assign_direction(tags: list[str]) -> str:
    """
    Assign trade direction from scanner tags.

    Rules evaluated in priority order — first match wins:
    1. Has pead-short OR rw-breakdown → short
    2. Has consecutive-miss AND no pead-long AND no ep-gap → short
    3. Has any long tag → long
    4. Has high-put-ratio only (no other tags) → long (direction-neutral default)
    5. Conflict (both long and short tags present) → long (lower-risk default)
    6. No tags → long (default)

    Parameters
    ----------
    tags:
        List of tag names assigned to the candidate.

    Returns
    -------
    str
        "long" or "short"
    """
    tag_set = set(tags)

    # Rule 5 (evaluated first): conflict between pead-long and pead-short → long
    # ScoringSystem.md: "Conflict (pead-long AND pead-short): direction = long"
    # Momentum asymmetry: long momentum is stronger and easier to execute than short.
    if "pead-long" in tag_set and "pead-short" in tag_set:
        return "long"

    # Rule 1: primary short signals (pead-short only when pead-long absent)
    if "pead-short" in tag_set or "rw-breakdown" in tag_set:
        return "short"

    # Rule 2: consecutive miss without offsetting long evidence
    if (
        "consecutive-miss" in tag_set
        and "pead-long" not in tag_set
        and "ep-gap" not in tag_set
    ):
        return "short"

    # Rules 3–6: long by default
    return "long"
