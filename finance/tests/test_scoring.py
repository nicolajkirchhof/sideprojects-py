"""Tests for the 5-box checklist scoring logic."""
from __future__ import annotations

import pytest

from finance.apps.analyst._models import (
    BoxResult,
    Candidate,
    EnrichedCandidate,
    ScoredCandidate,
    TechnicalData,
)
from finance.apps.analyst._scoring import score


def _make_enriched(
    symbol: str = "TEST",
    price: float = 100.0,
    technicals: TechnicalData | None = None,
    data_available: bool = True,
) -> EnrichedCandidate:
    return EnrichedCandidate(
        candidate=Candidate(symbol=symbol, price=price),
        technicals=technicals,
        data_available=data_available,
    )


def _ideal_technicals() -> TechnicalData:
    """Technicals that pass all boxes."""
    return TechnicalData(
        sma_5=102,
        sma_10=101,
        sma_20=99,
        sma_50=95,
        sma_200=85,
        sma_50_slope="rising",
        sma_200_slope="rising",
        bb_width=0.03,
        bb_width_avg_20=0.05,
        atr_14=2.5,
        high_52w=105,
        low_52w=60,
        return_12m=25.0,
        rs_vs_spy=1.2,
        rs_slope_10d=2.5,
        rvol=1.5,
        volume_contracting=True,
    )


class TestBox1TrendTemplate:
    def test_passes_with_proper_sma_stack(self) -> None:
        result = score([_make_enriched(technicals=_ideal_technicals())])
        box1 = result[0].boxes[0]
        assert box1.status == "PASS"
        assert "20 SMA" in box1.reason

    def test_fails_when_price_below_sma(self) -> None:
        t = _ideal_technicals()
        t.sma_20 = 105  # price 100 < sma_20 105
        result = score([_make_enriched(technicals=t)])
        box1 = result[0].boxes[0]
        assert box1.status == "FAIL"
        assert "stack broken" in box1.reason

    def test_fails_when_50sma_not_rising(self) -> None:
        t = _ideal_technicals()
        t.sma_50_slope = "falling"
        result = score([_make_enriched(technicals=t)])
        box1 = result[0].boxes[0]
        assert box1.status == "FAIL"
        assert "falling" in box1.reason

    def test_fails_negative_12m_return(self) -> None:
        t = _ideal_technicals()
        t.return_12m = -5.0
        result = score([_make_enriched(technicals=t)])
        box1 = result[0].boxes[0]
        assert box1.status == "FAIL"
        assert "negative" in box1.reason

    def test_fails_too_far_from_52w_high(self) -> None:
        t = _ideal_technicals()
        t.high_52w = 200  # price 100 is 50% below
        result = score([_make_enriched(technicals=t)])
        box1 = result[0].boxes[0]
        assert box1.status == "FAIL"
        assert "52W high" in box1.reason


class TestBox2RelativeStrength:
    def test_passes_with_positive_rs_slope(self) -> None:
        result = score([_make_enriched(technicals=_ideal_technicals())])
        box2 = result[0].boxes[1]
        assert box2.status == "PASS"
        assert "outperforming" in box2.reason

    def test_fails_with_negative_rs_slope(self) -> None:
        t = _ideal_technicals()
        t.rs_slope_10d = -1.5
        result = score([_make_enriched(technicals=t)])
        box2 = result[0].boxes[1]
        assert box2.status == "FAIL"
        assert "underperforming" in box2.reason

    def test_manual_when_no_rs_data(self) -> None:
        t = _ideal_technicals()
        t.rs_slope_10d = None
        result = score([_make_enriched(technicals=t)])
        box2 = result[0].boxes[1]
        assert box2.status == "MANUAL"


class TestBox3BaseQuality:
    def test_passes_with_squeeze_and_vdu(self) -> None:
        result = score([_make_enriched(technicals=_ideal_technicals())])
        box3 = result[0].boxes[2]
        assert box3.status == "PASS"
        assert "SMA stack intact" in box3.reason

    def test_fails_broken_sma_stack(self) -> None:
        t = _ideal_technicals()
        t.sma_5 = 90  # breaks 5 > 10 > 20 > 50
        result = score([_make_enriched(technicals=t)])
        box3 = result[0].boxes[2]
        assert box3.status == "FAIL"
        assert "stack broken" in box3.reason


class TestBox4Catalyst:
    def test_always_manual(self) -> None:
        result = score([_make_enriched(technicals=_ideal_technicals())])
        box4 = result[0].boxes[3]
        assert box4.status == "MANUAL"


class TestBox5Risk:
    def test_passes_with_tight_stop(self) -> None:
        # price 100, sma_20 99 → stop distance 1%
        result = score([_make_enriched(technicals=_ideal_technicals())])
        box5 = result[0].boxes[4]
        assert box5.status == "PASS"
        assert "within" in box5.reason

    def test_fails_stop_too_far(self) -> None:
        t = _ideal_technicals()
        t.sma_20 = 85  # 15% stop distance
        result = score([_make_enriched(price=100, technicals=t)])
        box5 = result[0].boxes[4]
        assert box5.status == "FAIL"
        assert "exceeds" in box5.reason

    def test_fails_price_below_sma(self) -> None:
        t = _ideal_technicals()
        t.sma_20 = 110  # price below stop
        result = score([_make_enriched(price=100, technicals=t)])
        box5 = result[0].boxes[4]
        assert box5.status == "FAIL"
        assert "below 20 SMA" in box5.reason


class TestNoData:
    def test_all_manual_when_no_data(self) -> None:
        ec = _make_enriched(data_available=False)
        result = score([ec])
        assert result[0].score == 0
        assert all(b.status == "MANUAL" for b in result[0].boxes)


class TestSorting:
    def test_sorted_by_score_then_rs(self) -> None:
        t_high = _ideal_technicals()
        t_high.rs_slope_10d = 5.0

        t_low = _ideal_technicals()
        t_low.sma_50_slope = "falling"  # fails box 1
        t_low.rs_slope_10d = 10.0  # higher RS but lower score

        candidates = [
            _make_enriched(symbol="LOW", technicals=t_low),
            _make_enriched(symbol="HIGH", technicals=t_high),
        ]
        result = score(candidates)
        # HIGH should be first (higher score despite lower RS)
        assert result[0].enriched.candidate.symbol == "HIGH"
