"""
Tests for the weighted scoring engine (finance.apps.assistant).

Tests follow TDD order: they define the expected behaviour of modules that
do not yet exist. Run them to see red before implementing the modules.

Modules under test:
  finance.apps.assistant._models  — ScoringConfig, ComponentScore, DimensionScore, CandidateScore
  finance.apps.assistant._scoring — _lerp, score_candidate
  finance.apps.assistant._tags    — assign_tags, assign_direction
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from finance.apps.analyst._config import load_config
from finance.apps.analyst._models import Candidate, EnrichedCandidate, TechnicalData
from finance.apps.analyst._scanner import parse_csv
from finance.apps.assistant._models import (
    CandidateScore,
    ComponentScore,
    DimensionScore,
    ScoringConfig,
)
from finance.apps.assistant._scoring import _lerp, score_candidate
from finance.apps.assistant._tags import (
    _parse_squeeze_state,
    assign_direction,
    assign_tags,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_SCANNER_DIR = Path(__file__).parent.parent / "_data" / "barchart" / "screener"

_SCANNER_FILES: dict[str, str] = {
    "long-universe": "stocks-screener-long-universe-04-22-2026.csv",
    "pead-scanner": "stocks-screener-pead-scanner-04-22-2026.csv",
    "ep-gap-scanner": "stocks-screener-ep-gap-scanner-04-22-2026.csv",
    "rw-breakdown-candidates": "stocks-screener-rw-breakdown-candidates-04-22-2026.csv",
    "short-squeeze": "stocks-screener-short-squeeze-04-22-2026.csv",
    "high-put-ratio": "stocks-screener-high-put-ratio-04-22-2026.csv",
    "high-call-ratio": "stocks-screener-high-call-ratio-04-22-2026.csv",
}

# Tags backed by scanner-file membership (not conditions on Candidate fields)
_MEMBERSHIP_SCANNER_TAGS: dict[str, str] = {
    "ep-gap-scanner": "ep-gap",
    "rw-breakdown-candidates": "rw-breakdown",
    "short-squeeze": "short-squeeze",
    "high-put-ratio": "high-put-ratio",
    "high-call-ratio": "high-call-ratio",
}


def _make_candidate(**kwargs) -> Candidate:
    defaults: dict = {
        "symbol": "TEST",
        "price": 100.0,
        "rvol_20d": 1.5,
        "atr_pct_20d": 4.0,
        "adr_pct_20d": 5.0,
        "pct_from_50d_sma": 5.0,          # 5% above 50d SMA
        "slope_50d_sma": 0.5,              # rising
        "slope_200d_sma": 0.3,             # rising
        "high_52w_distance_pct": -3.0,     # 3% below 52W high
        "change_52w_pct": 25.0,            # 12M return proxy
        "weighted_alpha": 30.0,
        "perf_vs_market_5d": 5.0,
        "perf_vs_market_1m": 8.0,
        "perf_vs_market_3m": 15.0,
        "put_call_vol_5d": 0.4,
        "earnings_surprise_pct": 12.0,
        "earnings_surprise_q1": 8.0,
        "earnings_surprise_q2": 5.0,
        "earnings_surprise_q3": 10.0,
        "latest_earnings": "2026-06-15",   # ~54 days out
        "market_cap_k": 5_000_000_000,     # $5B
        "ttm_squeeze": "On",
        "bb_pct": 75.0,
        "trend_seeker_signal": "Buy",
        "short_float": 5.0,
        "change_5d_pct": 8.0,
        "change_1m_pct": 15.0,
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


def _make_enriched(
    candidate: Candidate | None = None,
    technicals: TechnicalData | None = None,
    data_available: bool = True,
) -> EnrichedCandidate:
    c = candidate if candidate is not None else _make_candidate()
    return EnrichedCandidate(
        candidate=c,
        technicals=technicals,
        data_available=data_available,
    )


def _default_scoring_config() -> ScoringConfig:
    return ScoringConfig(
        weights={1: 25, 2: 25, 3: 15, 4: 20, 5: 15},
        tag_bonus_per_tag=2,
        tag_bonus_cap=12,
    )


def _ideal_technicals() -> TechnicalData:
    """IBKR-enriched data that produces high sub-component scores."""
    return TechnicalData(
        sma_5=106,
        sma_10=105,
        sma_20=104,
        sma_50=98,
        sma_200=85,
        sma_50_slope="rising",
        sma_200_slope="rising",
        bb_width=0.03,
        bb_width_avg_20=0.06,          # squeeze (width < avg)
        atr_14=4.0,
        high_52w=105,
        low_52w=60,
        return_12m=30.0,
        rs_vs_spy=1.25,
        rs_slope_10d=0.6,              # positive: outperforming SPY
        rvol=1.5,
        volume_contracting=True,
    )


# ---------------------------------------------------------------------------
# 1. _lerp — piecewise linear interpolation
# ---------------------------------------------------------------------------

class TestLerp:
    def test_exact_anchor_start(self) -> None:
        assert _lerp(0.0, [(0, 1.0), (5, 0.8), (15, 0.4)]) == pytest.approx(1.0)

    def test_exact_anchor_middle(self) -> None:
        assert _lerp(5.0, [(0, 1.0), (5, 0.8), (15, 0.4)]) == pytest.approx(0.8)

    def test_exact_anchor_end(self) -> None:
        assert _lerp(15.0, [(0, 1.0), (5, 0.8), (15, 0.4)]) == pytest.approx(0.4)

    def test_interpolation_between_anchors(self) -> None:
        # Midpoint between (0, 1.0) and (10, 0.0) → 0.5
        assert _lerp(5.0, [(0, 1.0), (10, 0.0)]) == pytest.approx(0.5)

    def test_clamp_below_first_anchor(self) -> None:
        # Below minimum x → clamp to first y
        assert _lerp(-5.0, [(0, 1.0), (10, 0.0)]) == pytest.approx(1.0)

    def test_clamp_above_last_anchor(self) -> None:
        # Above maximum x → clamp to last y
        assert _lerp(20.0, [(0, 1.0), (10, 0.0)]) == pytest.approx(0.0)

    def test_increasing_y_anchors(self) -> None:
        # 12M return: −20% → 0.0; 0% → 0.5; +20% → 1.0
        anchors = [(-20, 0.0), (0, 0.5), (20, 1.0)]
        assert _lerp(-20.0, anchors) == pytest.approx(0.0)
        assert _lerp(0.0, anchors) == pytest.approx(0.5)
        assert _lerp(20.0, anchors) == pytest.approx(1.0)
        assert _lerp(10.0, anchors) == pytest.approx(0.75)

    def test_result_clamped_to_zero_one(self) -> None:
        # Values outside anchor range do not exceed [0, 1]
        result = _lerp(100.0, [(0, 1.0), (10, 0.0)])
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 2. D1 — Trend Template (weight 25)
# ---------------------------------------------------------------------------

class TestD1TrendTemplate:
    def test_ideal_candidate_scores_full_weight(self) -> None:
        ec = _make_enriched(technicals=_ideal_technicals())
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d1 = result.dimensions[0]
        assert d1.dimension == 1
        # Perfect technicals: all 6 sub-components near 1.0 → weighted_score ≈ 25
        assert d1.weighted_score == pytest.approx(25.0, abs=2.0)

    def test_falling_50d_slope_penalises_score(self) -> None:
        t = _ideal_technicals()
        t.sma_50_slope = "falling"
        ec = _make_enriched(technicals=t)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d1 = result.dimensions[0]
        # 50d slope sub-component = 0.0 → D1 drops below 25
        assert d1.weighted_score < 25.0

    def test_far_from_52w_high_penalises_score(self) -> None:
        t = _ideal_technicals()
        t.high_52w = 200  # price 100 → 50% below high
        ec = _make_enriched(technicals=t)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d1 = result.dimensions[0]
        assert d1.weighted_score < 25.0

    def test_negative_12m_return_reduces_score(self) -> None:
        t = _ideal_technicals()
        t.return_12m = -10.0
        ec = _make_enriched(technicals=t)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d1 = result.dimensions[0]
        assert d1.weighted_score < 25.0

    def test_price_below_50d_sma_penalises(self) -> None:
        t = _ideal_technicals()
        t.sma_50 = 115  # price 100 is 13% below sma → price_vs_50d component = 0
        ec = _make_enriched(technicals=t)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d1_comp_score = result.dimensions[0].weighted_score
        # One of 6 components zeroed (price vs 50d SMA) → D1 drops below full weight
        assert d1_comp_score < 25.0
        # And clearly lower than the ideal score (all 1.0 → 25.0)
        assert d1_comp_score < 22.0

    def test_short_inversion_rewards_downtrend(self) -> None:
        """Stock with falling trend scores higher as short than as long."""
        t = _ideal_technicals()
        t.sma_50_slope = "falling"
        t.sma_200_slope = "falling"
        t.return_12m = -25.0
        t.high_52w = 200  # far from 52W high (bad for long, irrelevant for short)
        t.low_52w = 70    # close to 52W low (bad for long, good for short)
        ec = _make_enriched(technicals=t)
        score_long = score_candidate(ec, "long", [], _default_scoring_config())
        score_short = score_candidate(ec, "short", [], _default_scoring_config())
        d1_long = score_long.dimensions[0].weighted_score
        d1_short = score_short.dimensions[0].weighted_score
        assert d1_short > d1_long

    def test_scanner_fallback_when_no_ibkr_data(self) -> None:
        """When IBKR technicals unavailable, falls back to scanner fields."""
        c = _make_candidate(
            pct_from_50d_sma=3.0,
            slope_50d_sma=0.5,    # rising proxy
            slope_200d_sma=0.2,   # rising proxy
            high_52w_distance_pct=-4.0,
            weighted_alpha=25.0,
        )
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d1 = result.dimensions[0]
        # Score should be > 0 even without IBKR data
        assert d1.weighted_score > 0
        assert d1.partial is True  # flagged as partial (some components unavailable)

    def test_all_components_missing_gives_zero(self) -> None:
        """No IBKR data + no scanner fields → D1 = 0, flagged no data."""
        c = Candidate(symbol="BARE")  # all fields None
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d1 = result.dimensions[0]
        assert d1.weighted_score == pytest.approx(0.0)
        assert all(not comp.available for comp in d1.components)


# ---------------------------------------------------------------------------
# 3. D2 — Relative Strength (weight 25)
# ---------------------------------------------------------------------------

class TestD2RelativeStrength:
    def test_strong_outperformance_scores_high(self) -> None:
        c = _make_candidate(
            perf_vs_market_5d=6.0,    # > 5% threshold → 1.0
            perf_vs_market_1m=8.0,    # > 5% threshold → 1.0
            perf_vs_market_3m=12.0,   # > 10% threshold → 1.0
        )
        ec = _make_enriched(candidate=c, technicals=_ideal_technicals())
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d2 = result.dimensions[1]
        assert d2.weighted_score == pytest.approx(25.0, abs=3.0)

    def test_underperformance_scores_low(self) -> None:
        c = _make_candidate(
            perf_vs_market_5d=-6.0,
            perf_vs_market_1m=-6.0,
            perf_vs_market_3m=-12.0,
        )
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        # No RS slope (no IBKR) → only 3 scanner sub-components
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d2 = result.dimensions[1]
        assert d2.weighted_score < 10.0

    def test_short_inversion_rewards_underperformance(self) -> None:
        c = _make_candidate(
            perf_vs_market_5d=-6.0,
            perf_vs_market_1m=-8.0,
            perf_vs_market_3m=-15.0,
        )
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        score_long = score_candidate(ec, "long", [], _default_scoring_config())
        score_short = score_candidate(ec, "short", [], _default_scoring_config())
        d2_long = score_long.dimensions[1].weighted_score
        d2_short = score_short.dimensions[1].weighted_score
        assert d2_short > d2_long

    def test_rs_slope_excluded_when_no_ibkr(self) -> None:
        """RS slope sub-component unavailable without IBKR; other 3 still score."""
        c = _make_candidate(
            perf_vs_market_5d=5.0,
            perf_vs_market_1m=5.0,
            perf_vs_market_3m=10.0,
        )
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d2 = result.dimensions[1]
        # RS slope component should be unavailable
        rs_comp = next(c for c in d2.components if "rs_slope" in c.name.lower())
        assert not rs_comp.available
        # Other 3 components still available → weighted_score > 0
        assert d2.weighted_score > 0

    def test_rs_slope_positive_gives_high_score(self) -> None:
        t = _ideal_technicals()
        t.rs_slope_10d = 0.6   # ≥ 0.5%/day → 1.0
        ec = _make_enriched(technicals=t)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d2 = result.dimensions[1]
        rs_comp = next(c for c in d2.components if "rs_slope" in c.name.lower())
        assert rs_comp.available
        assert rs_comp.raw_score == pytest.approx(1.0)

    def test_rs_slope_negative_gives_zero(self) -> None:
        t = _ideal_technicals()
        t.rs_slope_10d = -0.5  # ≤ −0.5%/day → 0.0
        ec = _make_enriched(technicals=t)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d2 = result.dimensions[1]
        rs_comp = next(c for c in d2.components if "rs_slope" in c.name.lower())
        assert rs_comp.raw_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. D3 — Base Quality (weight 15)
# ---------------------------------------------------------------------------

class TestD3BaseQuality:
    def test_squeeze_on_scores_high_bb_contribution(self) -> None:
        """'On' state (building squeeze) → 0.8 for long."""
        c = _make_candidate(ttm_squeeze="On", bb_pct=40.0, adr_pct_20d=5.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        bb_comp = next(comp for comp in d3.components if "squeeze" in comp.name.lower() or "bb" in comp.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.8)

    def test_squeeze_off_wide_bb_scores_fired_proxy(self) -> None:
        """ttm_squeeze == "Off" AND bb_pct > 80 → 0.8 (proxy for recently fired)."""
        c = _make_candidate(ttm_squeeze="Off", bb_pct=90.0, adr_pct_20d=5.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        bb_comp = next(comp for comp in d3.components if "squeeze" in comp.name.lower() or "bb" in comp.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.8)

    def test_squeeze_off_narrow_bb_scores_low(self) -> None:
        c = _make_candidate(ttm_squeeze="Off", bb_pct=40.0, adr_pct_20d=5.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        bb_comp = next(comp for comp in d3.components if "squeeze" in comp.name.lower() or "bb" in comp.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.3)

    def test_squeeze_numeric_1_treated_as_on(self) -> None:
        """CSV stores TTM Squeeze as 0/1; "1" must be treated as On → 0.8 for long."""
        c = _make_candidate(ttm_squeeze="1", bb_pct=40.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        bb_comp = next(comp for comp in d3.components if "squeeze" in comp.name.lower() or "bb" in comp.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.8)

    def test_squeeze_numeric_0_treated_as_off(self) -> None:
        c = _make_candidate(ttm_squeeze="0", bb_pct=40.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        bb_comp = next(comp for comp in d3.components if "squeeze" in comp.name.lower() or "bb" in comp.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.3)

    def test_adr_in_ideal_range_scores_full(self) -> None:
        c = _make_candidate(adr_pct_20d=5.0)   # 3–7% range → 1.0
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        adr_comp = next(comp for comp in d3.components if "adr" in comp.name.lower())
        assert adr_comp.raw_score == pytest.approx(1.0)

    def test_adr_below_2_scores_minimum(self) -> None:
        c = _make_candidate(adr_pct_20d=1.0)   # < 2% → 0.2
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        adr_comp = next(comp for comp in d3.components if "adr" in comp.name.lower())
        assert adr_comp.raw_score == pytest.approx(0.2)

    def test_adr_above_10_scores_minimum(self) -> None:
        c = _make_candidate(adr_pct_20d=12.0)  # > 10% → 0.2
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        adr_comp = next(comp for comp in d3.components if "adr" in comp.name.lower())
        assert adr_comp.raw_score == pytest.approx(0.2)

    def test_short_high_rvol_rewarded_for_volume_component(self) -> None:
        """For shorts, high RVOL (distribution) should score well, unlike longs."""
        c_high_rvol = _make_candidate(rvol_20d=2.0)
        ec = _make_enriched(candidate=c_high_rvol, technicals=None, data_available=False)
        score_long = score_candidate(ec, "long", [], _default_scoring_config())
        score_short = score_candidate(ec, "short", [], _default_scoring_config())
        d3_long = score_long.dimensions[2]
        d3_short = score_short.dimensions[2]
        # For long, high RVOL in volume component is bad (no VDU); for short it's good
        vol_long = next(comp for comp in d3_long.components if "vol" in comp.name.lower())
        vol_short = next(comp for comp in d3_short.components if "vol" in comp.name.lower())
        assert vol_short.raw_score > vol_long.raw_score

    def test_ibkr_vdu_flag_takes_priority_for_long(self) -> None:
        """volume_contracting=True (IBKR VDU) → volume sub-component = 1.0 for long."""
        t = _ideal_technicals()
        t.volume_contracting = True
        ec = _make_enriched(technicals=t)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d3 = result.dimensions[2]
        vol_comp = next(comp for comp in d3.components if "vol" in comp.name.lower())
        assert vol_comp.raw_score == pytest.approx(1.0)
        assert vol_comp.source == "ibkr"

    # --- 4-state squeeze D3 scoring ---

    def test_fired_long_scores_full_for_long(self) -> None:
        c = _make_candidate(ttm_squeeze="Long", bb_pct=85.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        bb_comp = next(c for c in result.dimensions[2].components if "bb" in c.name.lower() or "squeeze" in c.name.lower())
        assert bb_comp.raw_score == pytest.approx(1.0)

    def test_fired_short_scores_full_for_short(self) -> None:
        c = _make_candidate(ttm_squeeze="Short", bb_pct=20.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "short", [], _default_scoring_config())
        bb_comp = next(c for c in result.dimensions[2].components if "bb" in c.name.lower() or "squeeze" in c.name.lower())
        assert bb_comp.raw_score == pytest.approx(1.0)

    def test_fired_long_scores_partial_for_short(self) -> None:
        """Fired in the wrong direction → 0.5."""
        c = _make_candidate(ttm_squeeze="Long", bb_pct=85.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "short", [], _default_scoring_config())
        bb_comp = next(c for c in result.dimensions[2].components if "bb" in c.name.lower() or "squeeze" in c.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.5)

    def test_squeeze_on_scores_0_8_for_long(self) -> None:
        """Squeeze building (not yet fired) → 0.8 for long (was 1.0 previously)."""
        c = _make_candidate(ttm_squeeze="On", bb_pct=40.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        bb_comp = next(c for c in result.dimensions[2].components if "bb" in c.name.lower() or "squeeze" in c.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.8)

    def test_na_squeeze_treated_as_off(self) -> None:
        """Barchart N/A → 'off' state; narrow BB → 0.3 for long."""
        c = _make_candidate(ttm_squeeze="N/A", bb_pct=40.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        bb_comp = next(c for c in result.dimensions[2].components if "bb" in c.name.lower() or "squeeze" in c.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.3)

    def test_na_squeeze_wide_bb_scores_fired_proxy(self) -> None:
        """N/A + bb_pct > 80 → still treated as recently-fired proxy → 0.8."""
        c = _make_candidate(ttm_squeeze="N/A", bb_pct=90.0)
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        bb_comp = next(c for c in result.dimensions[2].components if "bb" in c.name.lower() or "squeeze" in c.name.lower())
        assert bb_comp.raw_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 4a. _parse_squeeze_state unit tests
# ---------------------------------------------------------------------------

class TestParseSqueezeState:
    def test_on_returns_on(self) -> None:
        assert _parse_squeeze_state("On") == "on"
        assert _parse_squeeze_state("1") == "on"
        assert _parse_squeeze_state("true") == "on"

    def test_long_returns_fired_long(self) -> None:
        assert _parse_squeeze_state("Long") == "fired_long"
        assert _parse_squeeze_state("long") == "fired_long"

    def test_short_returns_fired_short(self) -> None:
        assert _parse_squeeze_state("Short") == "fired_short"
        assert _parse_squeeze_state("SHORT") == "fired_short"

    def test_off_returns_off(self) -> None:
        assert _parse_squeeze_state("Off") == "off"
        assert _parse_squeeze_state("0") == "off"
        assert _parse_squeeze_state("false") == "off"

    def test_na_returns_off(self) -> None:
        assert _parse_squeeze_state("N/A") == "off"
        assert _parse_squeeze_state("n/a") == "off"
        assert _parse_squeeze_state("na") == "off"

    def test_none_returns_none(self) -> None:
        assert _parse_squeeze_state(None) is None

    def test_unknown_returns_none(self) -> None:
        assert _parse_squeeze_state("garbage") is None

    def test_fired_long_satisfies_is_squeeze_on(self) -> None:
        """fired_long is truthy in callers that use _is_squeeze_on."""
        from finance.apps.assistant._tags import _is_squeeze_on
        assert _is_squeeze_on("Long") is True

    def test_fired_short_satisfies_is_squeeze_on(self) -> None:
        from finance.apps.assistant._tags import _is_squeeze_on
        assert _is_squeeze_on("Short") is True

    def test_ttm_fired_tag_on_fired_long(self) -> None:
        """Barchart 'Long' state + conditions → ttm-fired tag."""
        c = _make_candidate(ttm_squeeze="Long", bb_pct=85.0, rvol_20d=1.2, atr_pct_20d=5.0)
        tags = assign_tags(c, scanner_sets={})
        assert "ttm-fired" in tags

    def test_ttm_fired_tag_on_fired_short(self) -> None:
        """Barchart 'Short' state + conditions → ttm-fired tag."""
        c = _make_candidate(ttm_squeeze="Short", bb_pct=85.0, rvol_20d=1.2, atr_pct_20d=5.0)
        tags = assign_tags(c, scanner_sets={})
        assert "ttm-fired" in tags

    def test_ttm_fired_not_tagged_when_squeeze_on(self) -> None:
        """'On' state (not fired) should NOT trigger ttm-fired."""
        c = _make_candidate(ttm_squeeze="On", bb_pct=85.0, rvol_20d=1.2, atr_pct_20d=5.0)
        tags = assign_tags(c, scanner_sets={})
        assert "ttm-fired" not in tags

    def test_ttm_fired_not_tagged_when_na(self) -> None:
        """N/A state should NOT trigger ttm-fired (narrow BB)."""
        c = _make_candidate(ttm_squeeze="N/A", bb_pct=40.0, rvol_20d=1.2, atr_pct_20d=5.0)
        tags = assign_tags(c, scanner_sets={})
        assert "ttm-fired" not in tags

    def test_52w_high_tag_with_fired_long(self) -> None:
        """'Long' state satisfies _is_squeeze_on → triggers squeeze-based tags."""
        c = _make_candidate(
            high_52w_distance_pct=-2.0,
            ttm_squeeze="Long",
            rvol_20d=1.5,
        )
        tags = assign_tags(c, scanner_sets={})
        assert "52w-high" in tags


# ---------------------------------------------------------------------------
# 5. D4 — Catalyst (weight 20)
# ---------------------------------------------------------------------------

class TestD4Catalyst:
    def test_long_hard_gate_under_5_days_zeroes_d4(self) -> None:
        # 3 days from today — always within the 5-day long blackout
        in_3_days = (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")
        c = _make_candidate(latest_earnings=in_3_days)
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d4 = result.dimensions[3]
        assert d4.weighted_score == pytest.approx(0.0)
        assert d4.hard_gate_fired is True

    def test_short_hard_gate_under_10_days_zeroes_d4(self) -> None:
        # 7 days from today — within the 10-day short blackout but outside long blackout
        in_7_days = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
        c = _make_candidate(latest_earnings=in_7_days)
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "short", [], _default_scoring_config())
        d4 = result.dimensions[3]
        assert d4.weighted_score == pytest.approx(0.0)
        assert d4.hard_gate_fired is True

    def test_short_7_days_out_not_gated_for_long(self) -> None:
        # 7 days from today — outside the 5-day long blackout, so not gated for longs
        in_7_days = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
        c = _make_candidate(latest_earnings=in_7_days)
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d4 = result.dimensions[3]
        assert d4.hard_gate_fired is False
        assert d4.weighted_score > 0

    def test_strong_earnings_beat_scores_high_for_long(self) -> None:
        c = _make_candidate(
            earnings_surprise_pct=15.0,    # ≥ 10% → 1.0
            earnings_surprise_q1=8.0,      # all beats
            earnings_surprise_q2=5.0,
            earnings_surprise_q3=12.0,
            put_call_vol_5d=0.2,           # < 0.3 → 1.0
            rvol_20d=3.5,                  # ≥ 3.0 → 1.0
            latest_earnings="2026-06-15",
        )
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d4 = result.dimensions[3]
        assert d4.weighted_score == pytest.approx(20.0, abs=2.0)

    def test_earnings_miss_gives_zero_surprise_for_long(self) -> None:
        c = _make_candidate(earnings_surprise_pct=-5.0, latest_earnings="2026-06-15")
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d4 = result.dimensions[3]
        surprise_comp = next(c for c in d4.components if "surprise" in c.name.lower() and "history" not in c.name.lower())
        assert surprise_comp.raw_score == pytest.approx(0.0)

    def test_earnings_miss_gives_high_surprise_for_short(self) -> None:
        c = _make_candidate(earnings_surprise_pct=-12.0, latest_earnings="2026-06-15")
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "short", [], _default_scoring_config())
        d4 = result.dimensions[3]
        surprise_comp = next(c for c in d4.components if "surprise" in c.name.lower() and "history" not in c.name.lower())
        assert surprise_comp.raw_score == pytest.approx(1.0)

    def test_four_of_four_beats_scores_full_history(self) -> None:
        c = _make_candidate(
            earnings_surprise_pct=10.0,
            earnings_surprise_q1=8.0,
            earnings_surprise_q2=5.0,
            earnings_surprise_q3=12.0,    # 4/4 beats → 1.0
            latest_earnings="2026-06-15",
        )
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d4 = result.dimensions[3]
        hist_comp = next(c for c in d4.components if "history" in c.name.lower())
        assert hist_comp.raw_score == pytest.approx(1.0)

    def test_two_of_four_beats_scores_partial_history(self) -> None:
        c = _make_candidate(
            earnings_surprise_pct=8.0,
            earnings_surprise_q1=-2.0,    # miss
            earnings_surprise_q2=5.0,
            earnings_surprise_q3=-3.0,    # miss → 2/4 beats
            latest_earnings="2026-06-15",
        )
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d4 = result.dimensions[3]
        hist_comp = next(c for c in d4.components if "history" in c.name.lower())
        assert hist_comp.raw_score == pytest.approx(0.5)

    def test_earnings_date_unavailable_excludes_proximity(self) -> None:
        c = _make_candidate(latest_earnings=None)
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d4 = result.dimensions[3]
        prox_comp = next(c for c in d4.components if "proximity" in c.name.lower() or "earnings" in c.name.lower())
        assert not prox_comp.available


# ---------------------------------------------------------------------------
# 6. D5 — Risk (weight 15)
# ---------------------------------------------------------------------------

class TestD5Risk:
    def test_hard_gate_stop_over_7pct_zeroes_d5(self) -> None:
        c = _make_candidate(atr_pct_20d=8.0)  # > 7% → hard gate
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d5 = result.dimensions[4]
        assert d5.weighted_score == pytest.approx(0.0)
        assert d5.hard_gate_fired is True

    def test_stop_within_2pct_scores_full_stop_component(self) -> None:
        c = _make_candidate(atr_pct_20d=1.5, pct_from_50d_sma=1.5)  # tight stop
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d5 = result.dimensions[4]
        stop_comp = next(c for c in d5.components if "stop" in c.name.lower())
        assert stop_comp.raw_score == pytest.approx(1.0)

    def test_large_cap_scores_full_market_cap_component(self) -> None:
        c = _make_candidate(market_cap_k=50_000_000_000)  # $50B ≥ $10B → 1.0
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d5 = result.dimensions[4]
        cap_comp = next(c for c in d5.components if "cap" in c.name.lower())
        assert cap_comp.raw_score == pytest.approx(1.0)

    def test_small_cap_scores_partial_market_cap(self) -> None:
        c = _make_candidate(market_cap_k=300_000_000)  # $300M → between 0.3 and 0.5
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d5 = result.dimensions[4]
        cap_comp = next(c for c in d5.components if "cap" in c.name.lower())
        assert 0.3 <= cap_comp.raw_score <= 0.5

    def test_short_float_hard_gate_over_20pct(self) -> None:
        c = _make_candidate(short_float=25.0, atr_pct_20d=4.0)  # > 20% → hard gate for shorts
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "short", [], _default_scoring_config())
        d5 = result.dimensions[4]
        assert d5.weighted_score == pytest.approx(0.0)
        assert d5.hard_gate_fired is True

    def test_short_float_not_a_gate_for_longs(self) -> None:
        c = _make_candidate(short_float=25.0, atr_pct_20d=4.0)
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d5 = result.dimensions[4]
        # Short float gate only applies to shorts
        assert d5.hard_gate_fired is False
        assert d5.weighted_score > 0

    def test_short_float_component_absent_for_longs(self) -> None:
        """Short Float sub-component should not exist in D5 for long direction."""
        c = _make_candidate(short_float=15.0)
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d5 = result.dimensions[4]
        float_comp = [c for c in d5.components if "float" in c.name.lower() or "short_float" in c.name.lower()]
        assert len(float_comp) == 0

    def test_short_float_component_present_for_shorts(self) -> None:
        c = _make_candidate(short_float=10.0, atr_pct_20d=4.0)
        ec = _make_enriched(candidate=c)
        result = score_candidate(ec, "short", [], _default_scoring_config())
        d5 = result.dimensions[4]
        float_comp = [c for c in d5.components if "float" in c.name.lower() or "short_float" in c.name.lower()]
        assert len(float_comp) == 1


# ---------------------------------------------------------------------------
# 7. Tag assignment
# ---------------------------------------------------------------------------

class TestTagAssignment:
    def test_no_tags_for_bare_candidate(self) -> None:
        c = Candidate(symbol="BARE")
        tags = assign_tags(c, scanner_sets={})
        assert tags == []

    def test_vol_spike_tag(self) -> None:
        c = _make_candidate(rvol_20d=2.0)  # > 1.75
        tags = assign_tags(c, scanner_sets={})
        assert "vol-spike" in tags

    def test_vol_spike_not_assigned_below_threshold(self) -> None:
        c = _make_candidate(rvol_20d=1.5)  # ≤ 1.75
        tags = assign_tags(c, scanner_sets={})
        assert "vol-spike" not in tags

    def test_trend_seeker_tag(self) -> None:
        c = _make_candidate(trend_seeker_signal="Buy")
        tags = assign_tags(c, scanner_sets={})
        assert "trend-seeker" in tags

    def test_trend_seeker_not_assigned_for_sell(self) -> None:
        c = _make_candidate(trend_seeker_signal="Sell")
        tags = assign_tags(c, scanner_sets={})
        assert "trend-seeker" not in tags

    def test_52w_high_tag_all_conditions_met(self) -> None:
        c = _make_candidate(
            high_52w_distance_pct=-3.0,    # > -5% → within 5% of high
            ttm_squeeze="On",
            rvol_20d=1.2,
        )
        tags = assign_tags(c, scanner_sets={})
        assert "52w-high" in tags

    def test_52w_high_tag_fails_if_too_far_from_high(self) -> None:
        c = _make_candidate(
            high_52w_distance_pct=-10.0,   # < -5% → too far
            ttm_squeeze="On",
            rvol_20d=1.2,
        )
        tags = assign_tags(c, scanner_sets={})
        assert "52w-high" not in tags

    def test_5d_momentum_tag(self) -> None:
        c = _make_candidate(
            change_5d_pct=7.0,
            rvol_20d=1.5,
            perf_vs_market_5d=3.0,
        )
        tags = assign_tags(c, scanner_sets={})
        assert "5d-momentum" in tags

    def test_1m_strength_tag(self) -> None:
        c = _make_candidate(
            change_1m_pct=12.0,
            perf_vs_market_1m=5.0,
            ttm_squeeze="On",
        )
        tags = assign_tags(c, scanner_sets={})
        assert "1m-strength" in tags

    def test_ttm_fired_proxy_tag(self) -> None:
        c = _make_candidate(
            ttm_squeeze="Off",
            bb_pct=90.0,         # > 80 → BB expanded
            rvol_20d=1.2,
            atr_pct_20d=5.0,    # < 7%
        )
        tags = assign_tags(c, scanner_sets={})
        assert "ttm-fired" in tags

    def test_ttm_fired_not_tagged_when_atrp_too_high(self) -> None:
        c = _make_candidate(ttm_squeeze="Off", bb_pct=90.0, rvol_20d=1.2, atr_pct_20d=8.0)
        tags = assign_tags(c, scanner_sets={})
        assert "ttm-fired" not in tags

    def test_pead_long_tag(self) -> None:
        c = _make_candidate(
            earnings_surprise_pct=10.0,
            change_5d_pct=12.0,
            perf_vs_market_5d=5.0,
            weighted_alpha=20.0,
        )
        tags = assign_tags(c, scanner_sets={})
        assert "pead-long" in tags

    def test_pead_short_tag(self) -> None:
        c = _make_candidate(
            earnings_surprise_pct=-8.0,
            change_5d_pct=-7.0,
            pct_from_50d_sma=-5.0,     # below 50d SMA
            short_float=10.0,          # < 20%
        )
        tags = assign_tags(c, scanner_sets={})
        assert "pead-short" in tags

    def test_pead_short_blocked_by_high_short_float(self) -> None:
        c = _make_candidate(
            earnings_surprise_pct=-8.0,
            change_5d_pct=-7.0,
            pct_from_50d_sma=-5.0,
            short_float=25.0,          # ≥ 20% → blocked
        )
        tags = assign_tags(c, scanner_sets={})
        assert "pead-short" not in tags

    def test_consecutive_miss_tag(self) -> None:
        c = _make_candidate(
            earnings_surprise_pct=-3.0,    # current miss
            earnings_surprise_q1=-5.0,     # q1 miss
            earnings_surprise_q2=2.0,
            earnings_surprise_q3=-1.0,     # q3 miss → 2 of 3 prior < 0
        )
        tags = assign_tags(c, scanner_sets={})
        assert "consecutive-miss" in tags

    def test_consecutive_miss_not_tagged_with_only_one_prior_miss(self) -> None:
        c = _make_candidate(
            earnings_surprise_pct=-3.0,
            earnings_surprise_q1=-5.0,
            earnings_surprise_q2=2.0,
            earnings_surprise_q3=1.0,     # only 1 prior miss
        )
        tags = assign_tags(c, scanner_sets={})
        assert "consecutive-miss" not in tags

    def test_membership_ep_gap_tag(self) -> None:
        c = _make_candidate(symbol="MSFT")
        scanner_sets = {"ep-gap-scanner": {"AAPL", "MSFT", "TSLA"}}
        tags = assign_tags(c, scanner_sets=scanner_sets)
        assert "ep-gap" in tags

    def test_membership_rw_breakdown_tag(self) -> None:
        c = _make_candidate(symbol="GME")
        scanner_sets = {"rw-breakdown-candidates": {"GME", "AMC"}}
        tags = assign_tags(c, scanner_sets=scanner_sets)
        assert "rw-breakdown" in tags

    def test_membership_tag_not_assigned_if_symbol_absent(self) -> None:
        c = _make_candidate(symbol="AAPL")
        scanner_sets = {"ep-gap-scanner": {"MSFT", "TSLA"}}
        tags = assign_tags(c, scanner_sets=scanner_sets)
        assert "ep-gap" not in tags

    def test_multiple_tags_non_exclusive(self) -> None:
        """A single stock can carry multiple tags simultaneously."""
        c = _make_candidate(
            rvol_20d=2.0,                 # vol-spike
            trend_seeker_signal="Buy",    # trend-seeker
            change_5d_pct=7.0,
            perf_vs_market_5d=3.0,        # 5d-momentum
        )
        tags = assign_tags(c, scanner_sets={})
        assert "vol-spike" in tags
        assert "trend-seeker" in tags
        assert "5d-momentum" in tags

    def test_ttm_squeeze_numeric_1_treated_as_on_for_tags(self) -> None:
        """TTM Squeeze "1" from CSV should be treated the same as "On"."""
        c = _make_candidate(
            high_52w_distance_pct=-2.0,
            ttm_squeeze="1",     # numeric from CSV
            rvol_20d=1.5,
        )
        tags = assign_tags(c, scanner_sets={})
        assert "52w-high" in tags

    def test_ttm_squeeze_numeric_0_treated_as_off_for_tags(self) -> None:
        """TTM Squeeze "0" from CSV should NOT trigger squeeze-based tags."""
        c = _make_candidate(
            change_1m_pct=15.0,
            perf_vs_market_1m=8.0,
            ttm_squeeze="0",     # numeric from CSV → Off
        )
        tags = assign_tags(c, scanner_sets={})
        assert "1m-strength" not in tags  # requires squeeze On


# ---------------------------------------------------------------------------
# 8. Direction assignment
# ---------------------------------------------------------------------------

class TestDirectionAssignment:
    def test_pead_short_gives_short(self) -> None:
        assert assign_direction(["pead-short"]) == "short"

    def test_rw_breakdown_gives_short(self) -> None:
        assert assign_direction(["rw-breakdown"]) == "short"

    def test_consecutive_miss_alone_gives_short(self) -> None:
        assert assign_direction(["consecutive-miss"]) == "short"

    def test_consecutive_miss_with_pead_long_gives_long(self) -> None:
        # Rule 2: consecutive-miss AND pead-long → not short (long wins)
        assert assign_direction(["consecutive-miss", "pead-long"]) == "long"

    def test_consecutive_miss_with_ep_gap_gives_long(self) -> None:
        assert assign_direction(["consecutive-miss", "ep-gap"]) == "long"

    def test_long_tags_give_long(self) -> None:
        assert assign_direction(["52w-high", "vol-spike"]) == "long"
        assert assign_direction(["pead-long"]) == "long"
        assert assign_direction(["ep-gap"]) == "long"
        assert assign_direction(["high-call-ratio"]) == "long"

    def test_conflict_long_and_short_gives_long(self) -> None:
        # pead-long AND pead-short both present → long (lower-risk default)
        assert assign_direction(["pead-long", "pead-short"]) == "long"

    def test_no_tags_defaults_to_long(self) -> None:
        assert assign_direction([]) == "long"

    def test_high_put_ratio_alone_defaults_to_long(self) -> None:
        # high-put-ratio is direction-neutral; alone → long default
        assert assign_direction(["high-put-ratio"]) == "long"

    def test_high_put_ratio_with_short_tags_gives_short(self) -> None:
        # high-put-ratio amplifies direction; with rw-breakdown → short
        assert assign_direction(["high-put-ratio", "rw-breakdown"]) == "short"


# ---------------------------------------------------------------------------
# 9. Tag bonus
# ---------------------------------------------------------------------------

class TestTagBonus:
    def test_zero_tags_no_bonus(self) -> None:
        c = Candidate(symbol="BARE")
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        assert result.tag_bonus == pytest.approx(0.0)

    def test_single_tag_adds_2(self) -> None:
        ec = _make_enriched()
        result = score_candidate(ec, "long", ["vol-spike"], _default_scoring_config())
        assert result.tag_bonus == pytest.approx(2.0)

    def test_six_tags_adds_12(self) -> None:
        tags = ["vol-spike", "trend-seeker", "52w-high", "5d-momentum", "1m-strength", "pead-long"]
        ec = _make_enriched()
        result = score_candidate(ec, "long", tags, _default_scoring_config())
        assert result.tag_bonus == pytest.approx(12.0)

    def test_more_than_six_tags_capped_at_12(self) -> None:
        tags = ["vol-spike", "trend-seeker", "52w-high", "5d-momentum",
                "1m-strength", "pead-long", "ep-gap"]   # 7 tags → cap
        ec = _make_enriched()
        result = score_candidate(ec, "long", tags, _default_scoring_config())
        assert result.tag_bonus == pytest.approx(12.0)

    def test_total_includes_tag_bonus(self) -> None:
        tags = ["vol-spike"]
        ec = _make_enriched()
        result = score_candidate(ec, "long", tags, _default_scoring_config())
        # total = dimensions sum + bonus
        dim_total = sum(d.weighted_score for d in result.dimensions)
        assert result.total == pytest.approx(dim_total + 2.0)


# ---------------------------------------------------------------------------
# 10. Missing data policy
# ---------------------------------------------------------------------------

class TestMissingDataPolicy:
    def test_all_components_missing_gives_zero_weighted_score(self) -> None:
        c = Candidate(symbol="BARE")
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        for dim in result.dimensions:
            if not dim.hard_gate_fired:
                available = [comp for comp in dim.components if comp.available]
                if not available:
                    assert dim.weighted_score == pytest.approx(0.0)

    def test_partial_unavailable_excludes_from_average(self) -> None:
        """If some sub-components are None, score is based on available ones only."""
        # Supply only perf_vs_market_5d (one of 4 D2 sub-components)
        c = _make_candidate(
            perf_vs_market_5d=5.0,   # 1 component → raw_score = 1.0
            perf_vs_market_1m=None,  # unavailable
            perf_vs_market_3m=None,  # unavailable
        )
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d2 = result.dimensions[1]
        assert d2.partial is True
        # Should NOT score 0 (would be penalty for missing data)
        assert d2.weighted_score > 0

    def test_partial_flag_set_when_component_unavailable(self) -> None:
        c = _make_candidate(perf_vs_market_3m=None)  # one D2 component missing
        ec = _make_enriched(candidate=c, technicals=None, data_available=False)
        result = score_candidate(ec, "long", [], _default_scoring_config())
        d2 = result.dimensions[1]
        assert d2.partial is True


# ---------------------------------------------------------------------------
# 11. Score model properties
# ---------------------------------------------------------------------------

class TestCandidateScoreModel:
    def test_score_output_structure(self) -> None:
        ec = _make_enriched(technicals=_ideal_technicals())
        result = score_candidate(ec, "long", ["vol-spike"], _default_scoring_config())
        assert isinstance(result, CandidateScore)
        assert result.direction == "long"
        assert result.tags == ["vol-spike"]
        assert len(result.dimensions) == 5
        assert result.tag_bonus == pytest.approx(2.0)
        assert result.total == pytest.approx(
            sum(d.weighted_score for d in result.dimensions) + 2.0
        )

    def test_dimension_weights_match_config(self) -> None:
        ec = _make_enriched(technicals=_ideal_technicals())
        config = _default_scoring_config()
        result = score_candidate(ec, "long", [], config)
        for dim in result.dimensions:
            max_possible = config.weights[dim.dimension]
            assert dim.weighted_score <= max_possible + 0.01

    def test_total_never_exceeds_112(self) -> None:
        """Theoretical max is 100 + 12 bonus = 112."""
        tags = ["vol-spike", "trend-seeker", "52w-high", "5d-momentum",
                "1m-strength", "pead-long", "ep-gap"]
        ec = _make_enriched(technicals=_ideal_technicals())
        result = score_candidate(ec, "long", tags, _default_scoring_config())
        assert result.total <= 112.0 + 0.01


# ---------------------------------------------------------------------------
# 12. Integration — real CSV files
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _SCANNER_DIR.exists(),
    reason="Scanner CSV directory not present",
)
class TestIntegrationRealCSVs:
    @pytest.fixture(scope="class")
    def scanner_config(self):
        return load_config().scanner

    @pytest.fixture(scope="class")
    def all_candidates_by_scanner(self, scanner_config):
        result: dict[str, list[Candidate]] = {}
        for key, filename in _SCANNER_FILES.items():
            path = _SCANNER_DIR / filename
            if path.exists():
                candidates = parse_csv(path, scanner_config)
                result[key] = candidates
        return result

    @pytest.fixture(scope="class")
    def scanner_sets(self, all_candidates_by_scanner) -> dict[str, set[str]]:
        """Build membership sets for tag assignment."""
        sets: dict[str, set[str]] = {}
        for key in _MEMBERSHIP_SCANNER_TAGS:
            if key in all_candidates_by_scanner:
                sets[key] = {c.symbol for c in all_candidates_by_scanner[key]}
        return sets

    def test_long_universe_parsed_successfully(self, all_candidates_by_scanner) -> None:
        candidates = all_candidates_by_scanner.get("long-universe", [])
        assert len(candidates) > 100, "Long Universe scanner should have 100+ candidates"

    def test_all_scanners_parsed_successfully(self, all_candidates_by_scanner) -> None:
        for key in _SCANNER_FILES:
            assert key in all_candidates_by_scanner, f"Failed to parse: {key}"
            assert len(all_candidates_by_scanner[key]) > 0, f"No candidates in: {key}"

    def test_new_fields_parsed_from_long_universe(self, all_candidates_by_scanner) -> None:
        candidates = all_candidates_by_scanner["long-universe"]
        # Verify that new fields added to config.yaml are parsed
        c = candidates[0]
        assert c.trend_seeker_signal is not None, "trend_seeker_signal should be parsed"
        assert c.slope_50d_sma is not None, "slope_50d_sma should be parsed"
        assert c.slope_200d_sma is not None, "slope_200d_sma should be parsed"
        assert c.ttm_squeeze is not None, "ttm_squeeze should be parsed"
        assert c.market_cap_k is not None, "market_cap_k should be parsed"

    def test_new_fields_parsed_from_pead_scanner(self, all_candidates_by_scanner) -> None:
        candidates = all_candidates_by_scanner["pead-scanner"]
        c = candidates[0]
        assert c.earnings_surprise_pct is not None, "earnings_surprise_pct should be parsed"
        assert c.earnings_surprise_q1 is not None
        assert c.gap_up_pct is not None, "gap_up_pct should be parsed"
        assert c.short_float is not None, "short_float should be parsed"

    def test_new_fields_parsed_from_schema_c(self, all_candidates_by_scanner) -> None:
        candidates = all_candidates_by_scanner["high-put-ratio"]
        c = candidates[0]
        assert c.put_call_vol_1m is not None, "put_call_vol_1m should be parsed"
        assert c.iv_chg_5d is not None, "iv_chg_5d should be parsed"
        assert c.short_interest_k is not None, "short_interest_k should be parsed"
        assert c.vol_oi_ratio is not None, "vol_oi_ratio should be parsed"

    def test_tags_assigned_from_long_universe(self, all_candidates_by_scanner, scanner_sets) -> None:
        candidates = all_candidates_by_scanner["long-universe"]
        tagged = 0
        for c in candidates:
            tags = assign_tags(c, scanner_sets={})
            if tags:
                tagged += 1
        assert tagged > 0, "At least some long universe candidates should get tags"

    def test_membership_tags_assigned_from_scanner_sets(
        self, all_candidates_by_scanner, scanner_sets
    ) -> None:
        """Symbols appearing in ep-gap-scanner should get the ep-gap tag."""
        ep_gap_symbols = scanner_sets.get("ep-gap-scanner", set())
        if not ep_gap_symbols:
            pytest.skip("No ep-gap scanner candidates")
        # Take any ep-gap symbol and check it gets the tag via scanner_sets
        symbol = next(iter(ep_gap_symbols))
        c = Candidate(symbol=symbol)
        tags = assign_tags(c, scanner_sets=scanner_sets)
        assert "ep-gap" in tags

    def test_scores_in_valid_range(self, all_candidates_by_scanner, scanner_sets) -> None:
        config = _default_scoring_config()
        candidates = all_candidates_by_scanner["long-universe"]
        for candidate in candidates[:20]:  # spot-check first 20
            ec = EnrichedCandidate(candidate=candidate, technicals=None, data_available=False)
            tags = assign_tags(candidate, scanner_sets=scanner_sets)
            direction = assign_direction(tags)
            result = score_candidate(ec, direction, tags, config)
            assert 0.0 <= result.total <= 112.0, (
                f"{candidate.symbol}: score {result.total} out of valid range"
            )

    def test_short_candidates_identified(self, all_candidates_by_scanner, scanner_sets) -> None:
        """RW breakdown and PEAD short candidates should be assigned short direction."""
        rw_symbols = scanner_sets.get("rw-breakdown-candidates", set())
        if not rw_symbols:
            pytest.skip("No RW breakdown candidates")
        symbol = next(iter(rw_symbols))
        c = Candidate(symbol=symbol)
        tags = assign_tags(c, scanner_sets=scanner_sets)
        direction = assign_direction(tags)
        assert direction == "short"

    def test_no_scores_violate_dimension_weight_cap(self, all_candidates_by_scanner, scanner_sets) -> None:
        config = _default_scoring_config()
        candidates = all_candidates_by_scanner["long-universe"]
        for candidate in candidates[:50]:
            ec = EnrichedCandidate(candidate=candidate, technicals=None, data_available=False)
            tags = assign_tags(candidate, scanner_sets=scanner_sets)
            direction = assign_direction(tags)
            result = score_candidate(ec, direction, tags, config)
            for dim in result.dimensions:
                max_weight = config.weights[dim.dimension]
                assert dim.weighted_score <= max_weight + 0.01, (
                    f"{candidate.symbol} D{dim.dimension}: {dim.weighted_score} > {max_weight}"
                )
