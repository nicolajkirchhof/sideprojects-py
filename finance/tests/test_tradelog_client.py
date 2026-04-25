"""
Tests for TA-E7-S2 — Tradelog REST API client.

All external HTTP calls are mocked so tests run without a live server.
"""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from finance.apps.assistant._tradelog_client import (
    fetch_closed_trades,
    fetch_open_option_positions,
    fetch_open_stock_positions,
)

_BASE_URL = "http://localhost:5186"

_SAMPLE_TRADE = {
    "id": 1,
    "symbol": "AAPL",
    "direction": "Long",
    "openDate": "2026-03-01",
    "closeDate": "2026-03-15",
    "pnl": 250.0,
    "xAtrMove": 1.5,
}

_SAMPLE_OPTION = {
    "id": 10,
    "symbol": "SPY",
    "expiry": "2026-06-20",
    "delta": -0.25,
    "theta": 12.5,
    "iv": 18.3,
}

_SAMPLE_STOCK = {
    "id": 20,
    "symbol": "NVDA",
    "quantity": 100,
    "avgCost": 850.0,
    "currentPrice": 900.0,
    "unrealisedPnl": 5000.0,
}


def _mock_response(data: object, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp


# ---------------------------------------------------------------------------
# fetch_closed_trades
# ---------------------------------------------------------------------------

def test_fetch_closed_trades_returns_list():
    with patch("requests.get", return_value=_mock_response([_SAMPLE_TRADE])) as mock_get:
        result = fetch_closed_trades(_BASE_URL)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["symbol"] == "AAPL"
    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    assert "Closed" in str(call_kwargs)


def test_fetch_closed_trades_passes_since_param():
    with patch("requests.get", return_value=_mock_response([])) as mock_get:
        fetch_closed_trades(_BASE_URL, since=date(2026, 1, 1))
    params = mock_get.call_args.kwargs.get("params", {})
    assert params.get("since") == "2026-01-01"


def test_fetch_closed_trades_returns_empty_on_http_error():
    with patch("requests.get", return_value=_mock_response([], status_code=500)):
        result = fetch_closed_trades(_BASE_URL)
    assert result == []


def test_fetch_closed_trades_returns_empty_on_connection_error():
    from requests.exceptions import ConnectionError as ReqConnError
    with patch("requests.get", side_effect=ReqConnError("refused")):
        result = fetch_closed_trades(_BASE_URL)
    assert result == []


def test_fetch_closed_trades_returns_empty_on_timeout():
    from requests.exceptions import Timeout
    with patch("requests.get", side_effect=Timeout("timed out")):
        result = fetch_closed_trades(_BASE_URL)
    assert result == []


def test_fetch_closed_trades_returns_empty_for_non_list_response():
    with patch("requests.get", return_value=_mock_response({"error": "bad"})):
        result = fetch_closed_trades(_BASE_URL)
    assert result == []


# ---------------------------------------------------------------------------
# fetch_open_option_positions
# ---------------------------------------------------------------------------

def test_fetch_open_option_positions_returns_list():
    with patch("requests.get", return_value=_mock_response([_SAMPLE_OPTION])):
        result = fetch_open_option_positions(_BASE_URL)
    assert isinstance(result, list)
    assert result[0]["symbol"] == "SPY"


def test_fetch_open_option_positions_returns_empty_on_error():
    from requests.exceptions import ConnectionError as ReqConnError
    with patch("requests.get", side_effect=ReqConnError("refused")):
        result = fetch_open_option_positions(_BASE_URL)
    assert result == []


# ---------------------------------------------------------------------------
# fetch_open_stock_positions
# ---------------------------------------------------------------------------

def test_fetch_open_stock_positions_returns_list():
    with patch("requests.get", return_value=_mock_response([_SAMPLE_STOCK])):
        result = fetch_open_stock_positions(_BASE_URL)
    assert isinstance(result, list)
    assert result[0]["symbol"] == "NVDA"


def test_fetch_open_stock_positions_returns_empty_on_error():
    from requests.exceptions import ConnectionError as ReqConnError
    with patch("requests.get", side_effect=ReqConnError("refused")):
        result = fetch_open_stock_positions(_BASE_URL)
    assert result == []


def test_fetch_open_stock_positions_strips_trailing_slash():
    with patch("requests.get", return_value=_mock_response([])) as mock_get:
        fetch_open_stock_positions("http://localhost:5186/")
    url = mock_get.call_args.args[0]
    assert not url.endswith("//")
