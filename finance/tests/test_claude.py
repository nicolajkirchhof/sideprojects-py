"""Tests for Claude API client — prompt building and JSON parsing."""
from __future__ import annotations

import json

import pytest

from finance.apps.analyst._claude import (
    _format_candidates,
    _format_emails,
    _format_market_context,
    _format_trades,
    _parse_json,
)
from finance.apps.analyst._gmail import EmailMessage
from finance.apps.analyst._models import (
    BoxResult,
    Candidate,
    EnrichedCandidate,
    MarketSummary,
    ScoredCandidate,
    TechnicalData,
)
from datetime import datetime


class TestParseJson:
    def test_direct_json(self) -> None:
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_array(self) -> None:
        result = _parse_json('[{"a": 1}, {"b": 2}]')
        assert len(result) == 2

    def test_code_fence(self) -> None:
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = _parse_json(text)
        assert result == {"key": "value"}

    def test_embedded_json(self) -> None:
        text = 'Here is the analysis:\n{"regime": "GO", "themes": ["tech"]}\nEnd.'
        result = _parse_json(text)
        assert result["regime"] == "GO"

    def test_embedded_array(self) -> None:
        text = 'Results:\n[{"symbol": "AAPL"}]\nDone.'
        result = _parse_json(text)
        assert result[0]["symbol"] == "AAPL"

    def test_invalid_returns_none(self) -> None:
        assert _parse_json("not json at all") is None

    def test_empty_string(self) -> None:
        assert _parse_json("") is None


class TestFormatEmails:
    def test_formats_multiple_emails(self) -> None:
        emails = [
            EmailMessage(
                message_id="1", sender="test@example.com",
                subject="Morning Brief", date=datetime(2026, 4, 17),
                body_text="Markets opened higher today.",
            ),
            EmailMessage(
                message_id="2", sender="other@example.com",
                subject="Evening Wrap", date=datetime(2026, 4, 17),
                body_text="Markets closed at highs.",
            ),
        ]
        result = _format_emails(emails)
        assert "Morning Brief" in result
        assert "Evening Wrap" in result
        assert "---" in result  # separator

    def test_truncates_long_body(self) -> None:
        email = EmailMessage(
            message_id="1", sender="test@example.com",
            subject="Long", date=datetime(2026, 4, 17),
            body_text="x" * 5000,
        )
        result = _format_emails([email])
        assert len(result) < 5000


class TestFormatCandidates:
    def test_formats_with_technicals(self) -> None:
        sc = ScoredCandidate(
            enriched=EnrichedCandidate(
                candidate=Candidate(symbol="AAPL", price=185.50, sector="Technology"),
                technicals=TechnicalData(
                    sma_20=183.0, sma_50=175.0, sma_50_slope="rising",
                    rs_slope_10d=2.5, bb_width=0.03, bb_width_avg_20=0.05,
                    rvol=1.5, volume_contracting=True, atr_14=3.2, return_12m=25.0,
                ),
            ),
            boxes=[BoxResult(1, "Trend Template", "PASS", "All criteria met")],
            score=4,
        )
        result = _format_candidates([sc])
        assert "AAPL" in result
        assert "185.50" in result
        assert "rising" in result
        assert "BB squeeze: YES" in result

    def test_formats_without_technicals(self) -> None:
        sc = ScoredCandidate(
            enriched=EnrichedCandidate(
                candidate=Candidate(symbol="TSLA", price=250),
                data_available=False,
            ),
            boxes=[BoxResult(1, "Trend Template", "MANUAL", "No data")],
            score=0,
        )
        result = _format_candidates([sc])
        assert "TSLA" in result
        assert "250" in result


class TestFormatMarketContext:
    def test_formats_summary(self) -> None:
        summary = MarketSummary(
            regime="GO",
            regime_reasoning="SPY above all SMAs",
            themes=["Tech leadership"],
            risks=["FOMC next week"],
        )
        result = _format_market_context(summary)
        assert "GO" in result
        assert "Tech leadership" in result
        assert "FOMC" in result

    def test_empty_summary(self) -> None:
        result = _format_market_context(MarketSummary())
        assert "No market summary" in result


class TestFormatTrades:
    def test_formats_trade_dict(self) -> None:
        trades = [{
            "id": 42,
            "symbol": "NVDA",
            "date": "2026-04-01",
            "strategy": "Momentum",
            "typeOfTrade": "Long Call",
            "directional": "Bullish",
            "budget": "Speculative",
            "pnl": 1250.00,
            "status": "Closed",
            "notes": "Strong earnings beat, gap up on volume",
            "intendedManagement": "Close at 50% profit",
            "actualManagement": "Closed at 75% profit after 3 days",
        }]
        result = _format_trades(trades)
        assert "NVDA" in result
        assert "Momentum" in result
        assert "Long Call" in result
        assert "Strong earnings" in result
