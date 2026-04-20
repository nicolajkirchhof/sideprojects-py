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
    change_pct: float | None = None          # today's % change
    volume: float | None = None
    change_5d_pct: float | None = None
    change_1m_pct: float | None = None
    change_3m_pct: float | None = None
    change_6m_pct: float | None = None
    change_52w_pct: float | None = None      # 12-month return (century momentum)
    high_52w_distance_pct: float | None = None
    rvol_20d: float | None = None            # 20D relative volume
    atr_pct_20d: float | None = None         # 20D ATR as % of price
    pct_from_50d_sma: float | None = None    # distance from 50D SMA
    bb_pct: float | None = None              # Bollinger Band %
    put_call_vol_5d: float | None = None     # 5-day put/call volume ratio
    iv_percentile: float | None = None       # IV percentile (for options structure)
    short_interest_chg_pct: float | None = None
    days_to_cover: float | None = None
    market_cap_k: float | None = None        # market cap in thousands
    latest_earnings: str | None = None       # next earnings date
    sector: str | None = None


@dataclass
class OptionsContract:
    """A single unusual options activity contract from the options screener."""
    symbol: str                          # underlying symbol
    underlying_price: float | None = None
    iv_percentile: float | None = None
    implied_vol: float | None = None
    iv_chg_1d: float | None = None
    iv_chg_5d: float | None = None
    option_type: str = ""                # "Call" or "Put"
    strike: float | None = None
    expiration: str | None = None
    delta: float | None = None
    moneyness: str | None = None         # "ITM", "ATM", "OTM"
    vol_oi_ratio: float | None = None
    volume: float | None = None
    vol_pct_chg: float | None = None
    open_interest: float | None = None
    oi_pct_chg: float | None = None
    theta: float | None = None
    expires_before_earnings: str | None = None


@dataclass
class UoaSignal:
    """Aggregated unusual options activity signal per underlying."""
    symbol: str
    call_count: int = 0
    put_count: int = 0
    max_vol_oi: float = 0
    avg_delta_calls: float | None = None
    iv_percentile: float | None = None
    contracts: list[OptionsContract] = field(default_factory=list)

    @property
    def net_direction(self) -> str:
        if self.call_count > self.put_count:
            return "bullish"
        elif self.put_count > self.call_count:
            return "bearish"
        return "neutral"


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


@dataclass
class MarketSummary:
    """Claude-generated market context summary."""
    regime: str = ""           # GO | CAUTION | NO-GO
    regime_reasoning: str = ""
    themes: list[str] = field(default_factory=list)
    movers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    raw_response: str = ""     # full Claude response for audit


@dataclass
class TradeRecommendation:
    """Claude-generated trade recommendation for a single candidate."""
    symbol: str = ""
    setup_type: str = ""       # A | B | C | D | none
    profit_mechanism: str = "" # PM-01 through PM-05 | none
    thesis: str = ""
    catalyst_assessment: str = ""
    recommended_structure: str = ""
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    risk_reward: str = ""
    confidence: str = ""       # high | medium | low
    reasoning: str = ""


@dataclass
class TradeAnalysisResult:
    """Claude-generated compliance analysis for a single closed trade."""
    trade_id: int = 0
    symbol: str = ""
    score: int = 0             # 1-5 compliance score
    analysis: str = ""         # full markdown analysis


@dataclass
class ComplianceAggregate:
    """Aggregate insights across all reviewed trades."""
    avg_score: float = 0
    patterns: list[str] = field(default_factory=list)
    top_improvement: str = ""
    refinements: list[str] = field(default_factory=list)
