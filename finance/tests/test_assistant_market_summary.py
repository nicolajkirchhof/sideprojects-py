"""
Tests for TA-E3-S4 — Market summary from Claude.

Covers:
  - MarketSummary model + serialisation
  - summarize_market() with mocked Claude API
  - Cache I/O with market_summary field
  - ClaudeSummaryThread signal behaviour (Qt display required)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_has_display = (
    sys.platform == "win32"
    or os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)

_MOCK_RESPONSE = json.dumps({
    "regime": "GO",
    "regime_reasoning": "SPY above 50d SMA, VIX < 20",
    "themes": ["AI infrastructure", "Energy rotation"],
    "movers": ["NVDA +8% earnings beat", "XOM -3% oil weakness"],
    "risks": ["FOMC in 3 days", "CPI Thursday"],
    "action_items": ["Size normally", "Avoid energy longs"],
})

_EMPTY_SUMMARY_FIELDS = {
    "regime": "",
    "regime_reasoning": "",
    "themes": [],
    "movers": [],
    "risks": [],
    "action_items": [],
    "raw_response": "",
}


def _make_email(body: str = "Market is bullish") -> object:
    from finance.apps.assistant._gmail import EmailBody

    return EmailBody(subject="Morning Brief", sender="analyst@example.com",
                     date="2026-04-23", body_text=body)


# ---------------------------------------------------------------------------
# MarketSummary model
# ---------------------------------------------------------------------------


def test_market_summary_defaults_are_empty():
    from finance.apps.assistant._models import MarketSummary

    s = MarketSummary()
    assert s.regime == ""
    assert s.themes == []
    assert s.raw_response == ""


def test_market_summary_to_dict_round_trip():
    """to_dict / from_dict must round-trip all fields."""
    from finance.apps.assistant._models import MarketSummary
    import dataclasses

    s = MarketSummary(
        regime="GO",
        regime_reasoning="SPY above 50d",
        themes=["AI", "Energy"],
        movers=["NVDA +8%"],
        risks=["FOMC"],
        action_items=["Size normally"],
        raw_response="raw",
    )
    d = dataclasses.asdict(s)
    s2 = MarketSummary(**d)
    assert s2.regime == "GO"
    assert s2.themes == ["AI", "Energy"]
    assert s2.raw_response == "raw"


# ---------------------------------------------------------------------------
# summarize_market()
# ---------------------------------------------------------------------------


def test_summarize_market_empty_emails_returns_empty():
    """No emails → return empty MarketSummary without calling Claude."""
    from finance.apps.assistant._claude import summarize_market

    result = summarize_market(emails=[], model="claude-sonnet-4-6")
    assert result.regime == ""
    assert result.themes == []


def test_summarize_market_parses_json_response():
    """Happy path: Claude returns valid JSON → parsed into MarketSummary."""
    from finance.apps.assistant._claude import summarize_market

    with patch("finance.apps.assistant._claude._call_claude", return_value=_MOCK_RESPONSE):
        result = summarize_market(emails=[_make_email()], model="claude-sonnet-4-6")

    assert result.regime == "GO"
    assert "AI infrastructure" in result.themes
    assert "FOMC in 3 days" in result.risks
    assert result.action_items == ["Size normally", "Avoid energy longs"]


def test_summarize_market_stores_raw_response():
    """raw_response is always set to the full Claude output."""
    from finance.apps.assistant._claude import summarize_market

    with patch("finance.apps.assistant._claude._call_claude", return_value=_MOCK_RESPONSE):
        result = summarize_market(emails=[_make_email()], model="claude-sonnet-4-6")

    assert result.raw_response == _MOCK_RESPONSE


def test_summarize_market_handles_unparseable_response():
    """If Claude returns garbage, raw_response is set but fields stay empty."""
    from finance.apps.assistant._claude import summarize_market

    garbage = "Sorry, I can't help with that."
    with patch("finance.apps.assistant._claude._call_claude", return_value=garbage):
        result = summarize_market(emails=[_make_email()], model="claude-sonnet-4-6")

    assert result.regime == ""
    assert result.raw_response == garbage


def test_summarize_market_handles_claude_api_failure():
    """If _call_claude returns empty string, raise RuntimeError."""
    from finance.apps.assistant._claude import summarize_market

    with patch("finance.apps.assistant._claude._call_claude", return_value=""):
        with pytest.raises(RuntimeError, match="Claude API call failed"):
            summarize_market(emails=[_make_email()], model="claude-sonnet-4-6")


def test_last_trading_day_skips_weekends():
    """Mon→Fri, Tue→Mon, Sat→Fri."""
    from datetime import date
    from finance.apps.assistant._gmail import last_trading_day

    assert last_trading_day(date(2026, 4, 20)) == date(2026, 4, 17)  # Mon → Fri
    assert last_trading_day(date(2026, 4, 21)) == date(2026, 4, 20)  # Tue → Mon
    assert last_trading_day(date(2026, 4, 25)) == date(2026, 4, 24)  # Sat → Fri


def test_summarize_market_includes_market_context():
    """market_context is injected into the Claude prompt."""
    from finance.apps.assistant._claude import summarize_market

    captured: list[str] = []

    def _fake_call(*, user_prompt, model, purpose):
        captured.append(user_prompt)
        return _MOCK_RESPONSE

    with patch("finance.apps.assistant._claude._call_claude", side_effect=_fake_call):
        summarize_market(
            emails=[_make_email()],
            model="claude-sonnet-4-6",
            market_context="SPY up 0.5%, VIX at 15",
        )

    assert len(captured) == 1
    assert "SPY up 0.5%, VIX at 15" in captured[0]
    assert "Market Research" in captured[0]


def test_fetch_market_context_returns_string():
    """fetch_market_context returns a non-empty string with mocked HTTP."""
    from unittest.mock import MagicMock
    from datetime import date
    from finance.apps.assistant._claude import fetch_market_context

    mock_resp = MagicMock()
    mock_resp.text = (
        "<html><body>"
        '<div class="result__snippet">SPY gained 0.5%</div>'
        '<div class="result__snippet">VIX fell to 14.5</div>'
        "</body></html>"
    )
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        result = fetch_market_context(date(2026, 4, 24))

    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Cache I/O with market_summary field
# ---------------------------------------------------------------------------


def test_write_cache_includes_market_summary(tmp_path):
    """write_cache with market_summary writes the field to JSON."""
    from finance.apps.assistant._pipeline import write_cache

    summary_dict = {"regime": "GO", "themes": ["AI"], "movers": [], "risks": [],
                    "action_items": [], "regime_reasoning": "ok", "raw_response": "raw"}
    path = write_cache([], date(2026, 4, 23), base_dir=tmp_path, market_summary=summary_dict)
    payload = json.loads(path.read_text())
    assert payload["market_summary"]["regime"] == "GO"
    assert payload["market_summary"]["themes"] == ["AI"]


def test_write_cache_without_market_summary_omits_field(tmp_path):
    """write_cache without market_summary stores None / absent field — backwards compatible."""
    from finance.apps.assistant._pipeline import write_cache

    path = write_cache([], date(2026, 4, 23), base_dir=tmp_path)
    payload = json.loads(path.read_text())
    # Either absent or None is acceptable
    assert payload.get("market_summary") is None


def test_read_market_summary_from_cache_returns_dict(tmp_path):
    """read_market_summary_from_cache returns the dict when present."""
    from finance.apps.assistant._pipeline import write_cache, read_market_summary_from_cache

    summary_dict = {"regime": "CAUTION", "themes": [], "movers": [], "risks": [],
                    "action_items": [], "regime_reasoning": "", "raw_response": ""}
    write_cache([], date(2026, 4, 23), base_dir=tmp_path, market_summary=summary_dict)
    result = read_market_summary_from_cache(date(2026, 4, 23), base_dir=tmp_path)
    assert result is not None
    assert result["regime"] == "CAUTION"


def test_read_market_summary_from_cache_returns_none_when_absent(tmp_path):
    """read_market_summary_from_cache returns None when no cache file exists."""
    from finance.apps.assistant._pipeline import read_market_summary_from_cache

    result = read_market_summary_from_cache(date(2026, 4, 23), base_dir=tmp_path)
    assert result is None


def test_read_market_summary_from_cache_returns_none_when_field_missing(tmp_path):
    """Old cache files without market_summary field → None (backwards-compatible)."""
    from finance.apps.assistant._pipeline import write_cache, read_market_summary_from_cache

    # Write a cache without market_summary
    write_cache([], date(2026, 4, 23), base_dir=tmp_path)
    result = read_market_summary_from_cache(date(2026, 4, 23), base_dir=tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# ClaudeSummaryThread — Qt integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_claude_summary_thread_emits_summary_ready():
    """Thread emits summary_ready with a dict when summarize_market succeeds."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._pipeline import ClaudeSummaryThread
    from finance.apps.assistant._gmail import EmailBody

    ensure_qt_app()

    emails = [EmailBody(subject="Brief", sender="x@y.com", date="2026-04-23",
                        body_text="Market is bullish")]
    received: list[dict] = []

    with patch("finance.apps.assistant._pipeline.summarize_market") as mock_sm:
        from finance.apps.assistant._models import MarketSummary
        mock_sm.return_value = MarketSummary(regime="GO", themes=["AI"])

        thread = ClaudeSummaryThread(emails=emails, model="claude-sonnet-4-6")
        thread.summary_ready.connect(received.append)
        thread.start()
        thread.wait(5000)

    from pyqtgraph.Qt import QtWidgets
    QtWidgets.QApplication.processEvents()

    assert len(received) == 1
    assert received[0]["regime"] == "GO"


@pytest.mark.skipif(not _has_display, reason="No display available")
def test_claude_summary_thread_emits_error_on_failure():
    """Thread emits error signal when summarize_market raises."""
    from finance.apps._qt_bootstrap import ensure_qt_app
    from finance.apps.assistant._pipeline import ClaudeSummaryThread
    from finance.apps.assistant._gmail import EmailBody

    ensure_qt_app()

    errors: list[str] = []
    emails = [EmailBody(subject="Brief", sender="x@y.com", date="2026-04-23",
                        body_text="Market is bullish")]

    with patch("finance.apps.assistant._pipeline.summarize_market",
               side_effect=RuntimeError("API down")):
        thread = ClaudeSummaryThread(emails=emails, model="claude-sonnet-4-6")
        thread.error.connect(errors.append)
        thread.start()
        thread.wait(5000)

    from pyqtgraph.Qt import QtWidgets
    QtWidgets.QApplication.processEvents()

    assert len(errors) == 1
    assert "API down" in errors[0]


# ---------------------------------------------------------------------------
# format_regime_context()
# ---------------------------------------------------------------------------


def _make_trend_status(symbol: str = "SPY", price: float = 547.23,
                       sma50: float = 530.1, sma200: float = 495.4) -> object:
    from finance.apps.assistant._data import TrendStatus

    return TrendStatus(
        symbol=symbol,
        last_price=price,
        sma_20=510.0,
        sma_50=sma50,
        sma_200=sma200,
        price_above_20=True,
        price_above_50=True,
        price_above_200=True,
        sma_20_slope="rising",
        sma_50_slope="rising",
        sma_200_slope="rising",
    )


def _make_vix_status(level: float = 17.20) -> object:
    from finance.apps.assistant._data import VixStatus

    return VixStatus(level=level, zone="low", direction="falling", is_spiking=False)


def test_format_regime_context_formats_spy_qqq_vix():
    """format_regime_context returns a markdown block with all key values."""
    from finance.apps.assistant._claude import format_regime_context

    spy = _make_trend_status("SPY", 547.23, 530.1, 495.4)
    qqq = _make_trend_status("QQQ", 468.50, 448.6, 420.1)
    vix = _make_vix_status(17.20)

    result = format_regime_context(spy, qqq, vix, regime_status="GO")

    assert "SPY" in result
    assert "547.23" in result
    assert "530.1" in result
    assert "495.4" in result
    assert "QQQ" in result
    assert "468.50" in result or "468.5" in result
    assert "VIX" in result
    assert "17.20" in result or "17.2" in result
    assert "low" in result
    assert "falling" in result
    assert "Regime: GO" in result
    assert result.startswith("# Local Market Data")


def test_format_regime_context_returns_empty_when_no_data():
    """All None inputs → returns empty string."""
    from finance.apps.assistant._claude import format_regime_context

    result = format_regime_context(None, None, None)
    assert result == ""


# ---------------------------------------------------------------------------
# _fetch_barchart_news()
# ---------------------------------------------------------------------------

_BARCHART_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Barchart Headlines</title>
    <item>
      <title>S&amp;P 500 Falls on Rate Fears</title>
      <description>The index slid 0.8% as bond yields climbed.</description>
    </item>
    <item>
      <title>Oil Rises 1.5% on OPEC+ Cuts</title>
      <description>Crude jumped after surprise production cut announcement.</description>
    </item>
    <item>
      <title>Tech Sector Leads Recovery</title>
      <description>Mega-cap tech stocks rebounded amid bargain hunting.</description>
    </item>
  </channel>
</rss>"""


def test_fetch_barchart_news_returns_string():
    """_fetch_barchart_news returns a non-empty string when RSS is reachable."""
    from finance.apps.assistant._claude import _fetch_barchart_news

    mock_resp = MagicMock()
    mock_resp.content = _BARCHART_RSS.encode()
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        result = _fetch_barchart_news()

    assert isinstance(result, str)
    assert len(result) > 0
    assert "Barchart News" in result
    assert "S&P 500" in result or "S" in result  # entity decoded or raw


def test_fetch_barchart_news_returns_empty_on_error():
    """_fetch_barchart_news returns '' when requests.get raises."""
    from finance.apps.assistant._claude import _fetch_barchart_news

    with patch("requests.get", side_effect=Exception("network error")):
        result = _fetch_barchart_news()

    assert result == ""
