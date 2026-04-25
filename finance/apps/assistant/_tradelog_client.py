"""
finance.apps.assistant._tradelog_client
=========================================
Best-effort REST client for the Tradelog API (TA-E7-S2).

All functions return [] on any error so callers are never blocked by an
unavailable server.  The Tradelog backend runs at http://localhost:5186
(configured in config.yaml under tradelog.base_url).
"""
from __future__ import annotations

import logging
from datetime import date

log = logging.getLogger(__name__)


def fetch_closed_trades(
    base_url: str,
    *,
    since: date | None = None,
    timeout: int = 10,
) -> list[dict]:
    """
    Fetch closed trades from the Tradelog API.

    Parameters
    ----------
    base_url:
        Root URL, e.g. ``"http://localhost:5186"``.
    since:
        Optional filter — only return trades closed on or after this date.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    list[dict]
        One dict per trade; empty list on any error.
    """
    try:
        import requests
    except ImportError:
        log.error("requests package not installed")
        return []

    url = f"{base_url.rstrip('/')}/api/trades/export"
    params: dict = {"status": "Closed"}
    if since is not None:
        params["since"] = since.isoformat()

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        log.warning("Unexpected response shape from %s: %r", url, type(data))
        return []
    except Exception:
        log.warning("fetch_closed_trades failed for %s", base_url, exc_info=True)
        return []


def fetch_open_option_positions(
    base_url: str,
    *,
    timeout: int = 10,
) -> list[dict]:
    """
    Fetch open option positions from the Tradelog API.

    Returns
    -------
    list[dict]
        One dict per position; empty list on any error.
    """
    try:
        import requests
    except ImportError:
        log.error("requests package not installed")
        return []

    url = f"{base_url.rstrip('/')}/api/option-positions"
    params: dict = {"status": "open"}

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        log.warning("Unexpected response shape from %s: %r", url, type(data))
        return []
    except Exception:
        log.warning("fetch_open_option_positions failed for %s", base_url, exc_info=True)
        return []


def fetch_open_stock_positions(
    base_url: str,
    *,
    timeout: int = 10,
) -> list[dict]:
    """
    Fetch open stock positions from the Tradelog API.

    Returns
    -------
    list[dict]
        One dict per position; empty list on any error.
    """
    try:
        import requests
    except ImportError:
        log.error("requests package not installed")
        return []

    url = f"{base_url.rstrip('/')}/api/stock-positions"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        log.warning("Unexpected response shape from %s: %r", url, type(data))
        return []
    except Exception:
        log.warning("fetch_open_stock_positions failed for %s", base_url, exc_info=True)
        return []
