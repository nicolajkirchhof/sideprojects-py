"""
finance.apps.assistant._models
===============================
Data classes for the weighted scoring engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoringConfig:
    """
    Configuration for the weighted scoring engine.

    weights: per-dimension weight (must sum to 100).
    tag_bonus_per_tag: bonus points per scanner tag.
    tag_bonus_cap: maximum total tag bonus.
    slope_flat_threshold: absolute slope value below which a SMA slope is
        classified as "flat" when only a numeric scanner value is available.
    """
    weights: dict[int, int] = field(
        default_factory=lambda: {1: 25, 2: 25, 3: 15, 4: 20, 5: 15}
    )
    tag_bonus_per_tag: int = 2
    tag_bonus_cap: int = 12
    slope_flat_threshold: float = 0.1


@dataclass
class ComponentScore:
    """Result for a single sub-component within a dimension."""
    name: str
    raw_score: float       # 0.0–1.0
    available: bool        # False when the data source was absent
    source: str            # "ibkr" | "scanner" | "none"


@dataclass
class DimensionScore:
    """Scoring result for one of the five scoring dimensions."""
    dimension: int          # 1–5
    name: str
    raw_score: float        # 0.0–1.0 (weighted average of available components)
    weighted_score: float   # raw_score × weight (0–weight), or 0 if hard gate fired
    components: list[ComponentScore] = field(default_factory=list)
    hard_gate_fired: bool = False
    partial: bool = False   # True if any component was unavailable


@dataclass
class CandidateScore:
    """Full scoring result for a single candidate."""
    direction: str                             # "long" | "short"
    dimensions: list[DimensionScore] = field(default_factory=list)
    tag_bonus: float = 0.0
    total: float = 0.0                         # sum of weighted_scores + tag_bonus
    tags: list[str] = field(default_factory=list)


@dataclass
class MarketSummary:
    """Claude-generated market context summary for the evening prep session."""
    regime: str = ""                    # "GO" | "CAUTION" | "NO-GO" | ""
    regime_reasoning: str = ""
    themes: list[str] = field(default_factory=list)
    movers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    raw_response: str = ""              # full Claude response for audit


@dataclass
class CandidateAnalysis:
    """Claude-generated trade analysis for a single watchlist candidate."""
    setup_type: str = ""                # e.g. "Type A — EP"
    profit_mechanism: str = ""          # e.g. "PM-02 PEAD"
    thesis: str = ""                    # 1-2 sentence trade thesis
    entry: float | None = None          # suggested entry price
    stop: float | None = None           # suggested stop price
    target: float | None = None         # suggested target price
    confidence: str = ""                # "LOW" | "MEDIUM" | "HIGH"
    raw_response: str = ""              # full Claude response for audit
