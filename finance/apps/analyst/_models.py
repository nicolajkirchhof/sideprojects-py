"""
finance.apps.analyst._models
=============================
Data classes for the analyst pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """Raw scanner candidate after CSV parsing and deduplication."""
    symbol: str
    price: float | None = None
    volume: float | None = None
    change_5d_pct: float | None = None
    change_1m_pct: float | None = None
    high_52w_distance_pct: float | None = None
    sector: str | None = None


@dataclass
class TechnicalData:
    """Technical indicators computed from IBKR daily data."""
    sma_5: float | None = None
    sma_10: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    sma_50_slope: str | None = None   # "rising" | "flat" | "falling"
    sma_200_slope: str | None = None
    bb_width: float | None = None
    bb_width_avg_20: float | None = None
    atr_14: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    return_12m: float | None = None
    rs_vs_spy: float | None = None        # relative strength ratio
    rs_slope_10d: float | None = None     # slope of RS line over 10 days
    rvol: float | None = None             # current vol / 50d avg vol
    volume_contracting: bool | None = None  # VDU detection


@dataclass
class BoxResult:
    """Result of a single checklist box evaluation."""
    box: int
    name: str
    status: str  # "PASS" | "FAIL" | "MANUAL"
    reason: str


@dataclass
class EnrichedCandidate:
    """Scanner candidate enriched with technical data."""
    candidate: Candidate
    technicals: TechnicalData | None = None
    data_available: bool = True


@dataclass
class ScoredCandidate:
    """Candidate with 5-box scoring results."""
    enriched: EnrichedCandidate
    boxes: list[BoxResult] = field(default_factory=list)
    score: int = 0  # count of PASS boxes (0-5)
