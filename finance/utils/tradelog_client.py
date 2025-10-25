"""
Tradelog REST client

A lightweight Python client to interact with the Tradelog backend (.NET) via its REST API.

Endpoints supported (as implemented in backend controllers):
- Instruments: /api/Instruments
- Logs:        /api/Logs
- Positions:   /api/Positions

Quick start:
    from finance.utils.tradelog_client import TradelogClient, InstrumentIn, LogIn, PositionIn

    client = TradelogClient(base_url="https://localhost:5001", verify_ssl=False)

    # Create or get an instrument
    inst = client.ensure_instrument({
        "SecType": "Stock",
        "Symbol": "AAPL",
        "Multiplier": 1,
        "Sector": "Tech",
        "Subsector": "Consumer Electronics",
    })

    # Add a log
    client.create_log({
        "InstrumentId": inst["Id"],
        "Date": "2025-10-25T00:00:00Z",
        "Notes": "Broke out of range",
        "ProfitMechanism": 1,  # Momentum flag
        "Sentiment": 1,        # Bullish flag
    })

    # Create a position
    client.create_position({
        "InstrumentId": inst["Id"],
        "Instrument": None,  # server can ignore or derive
        "InstrumentSpecifics": None,
        "ContractId": "OPT:12345678",
        "Type": "Call",
        "Opened": "2025-10-25T14:30:00Z",
        "Expiry": "2025-11-15T14:30:00Z",
        "Closed": None,
        "Size": 1,
        "Strike": 200.0,
        "Cost": 150.0,
        "Close": None,
        "Comission": 0.65,
        "CloseReasons": None,
    })

Notes:
- HTTPS redirection is enabled in the backend; depending on your hosting profile, base_url might be
  e.g. https://localhost:5001, https://localhost:7147, or similar.
- Enum-like fields are serialized as strings in the backend for enum-typed properties (e.g. Instrument.SecType,
  Position.Type) because JsonStringEnumConverter is enabled. Flag aggregations stored as integers (e.g. Log.Sentiment,
  Log.ProfitMechanism and Position.CloseReasons) should be sent as ints (bitmasks) or null.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, TypedDict, Union
import json
import time

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


# ----- Types mirroring backend models -----
SecType = Literal["Stock", "Future", "Forex"]
PositionType = Literal["Call", "Put", "Underlying"]


class InstrumentIn(TypedDict, total=False):
    SecType: SecType
    Symbol: str
    Multiplier: int
    Sector: str
    Subsector: Optional[str]


class Instrument(InstrumentIn, total=False):
    Id: int


class LogIn(TypedDict, total=False):
    InstrumentId: int
    Date: str  # ISO 8601
    Notes: Optional[str]
    ProfitMechanism: Optional[int]  # bit flags, nullable
    Sentiment: Optional[int]        # bit flags, nullable


class Log(LogIn, total=False):
    Id: int


class PositionIn(TypedDict, total=False):
    InstrumentId: int
    Instrument: Optional[dict]
    InstrumentSpecifics: Optional[str]
    ContractId: str
    Type: PositionType
    Opened: str  # ISO 8601
    Expiry: str  # ISO 8601
    Closed: Optional[str]
    Size: int
    Strike: float
    Cost: float
    Close: Optional[float]
    Comission: Optional[float]
    CloseReasons: Optional[int]


class Position(PositionIn, total=False):
    Id: int


# ----- Errors -----
class TradelogError(Exception):
    pass


@dataclass
class _RetryConfig:
    retries: int = 3
    backoff_factor: float = 0.3
    status_forcelist: Iterable[int] = (500, 502, 503, 504)


class TradelogClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        verify_ssl: bool = True,
        retry: Optional[_RetryConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Create a client for the Tradelog REST API.

        Args:
            base_url: Base address of the backend (e.g., "https://localhost:5001").
            timeout: Default request timeout in seconds.
            verify_ssl: When False, skip TLS verification (useful for dev certs).
            retry: Optional retry configuration for transient HTTP errors.
            session: Optional pre-configured requests.Session.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session = session or requests.Session()
        self._install_retries(retry or _RetryConfig())

    # ----- Public: Instruments -----
    def list_instruments(self) -> List[Instrument]:
        return self._get_json("/api/Instruments")

    def get_instrument(self, id: int) -> Instrument:
        return self._get_json(f"/api/Instruments/{id}")

    def create_instrument(self, instrument: InstrumentIn) -> Instrument:
        return self._post_json("/api/Instruments", instrument)

    def update_instrument(self, id: int, instrument: InstrumentIn) -> None:
        self._put_json(f"/api/Instruments/{id}", {**instrument, "Id": id})

    def delete_instrument(self, id: int) -> None:
        self._delete(f"/api/Instruments/{id}")

    def find_instrument_by_symbol(self, symbol: str) -> Optional[Instrument]:
        """Convenience helper: find first instrument with the given symbol (case-sensitive)."""
        for inst in self.list_instruments():
            if inst.get("Symbol") == symbol:
                return inst
        return None

    def ensure_instrument(self, instrument: InstrumentIn) -> Instrument:
        """Get or create an instrument by its Symbol (and optionally SecType)."""
        sym = instrument.get("Symbol")
        if not sym:
            raise ValueError("InstrumentIn requires 'Symbol' to ensure existence")
        existing = self.find_instrument_by_symbol(sym)
        return existing or self.create_instrument(instrument)

    # ----- Public: Logs -----
    def list_logs(self) -> List[Log]:
        return self._get_json("/api/Logs")

    def get_log(self, id: int) -> Log:
        return self._get_json(f"/api/Logs/{id}")

    def create_log(self, log: LogIn) -> Log:
        return self._post_json("/api/Logs", log)

    def update_log(self, id: int, log: LogIn) -> None:
        self._put_json(f"/api/Logs/{id}", {**log, "Id": id})

    def delete_log(self, id: int) -> None:
        self._delete(f"/api/Logs/{id}")

    # ----- Public: Positions -----
    def list_positions(self) -> List[Position]:
        return self._get_json("/api/Positions")

    def get_position(self, id: int) -> Position:
        return self._get_json(f"/api/Positions/{id}")

    def create_position(self, position: PositionIn) -> Position:
        return self._post_json("/api/Positions", position)

    def update_position(self, id: int, position: PositionIn) -> None:
        self._put_json(f"/api/Positions/{id}", {**position, "Id": id})

    def delete_position(self, id: int) -> None:
        self._delete(f"/api/Positions/{id}")

    # ----- Internal helpers -----
    def _install_retries(self, cfg: _RetryConfig) -> None:
        retry = Retry(
            total=cfg.retries,
            read=cfg.retries,
            connect=cfg.retries,
            backoff_factor=cfg.backoff_factor,
            status_forcelist=list(cfg.status_forcelist),
            allowed_methods=("GET", "POST", "PUT", "DELETE"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def _handle_error(self, resp: Response) -> None:
        try:
            data = resp.json()
        except Exception:
            data = None
        msg = None
        if isinstance(data, dict):
            msg = data.get("title") or data.get("detail") or data.get("error")
        if not msg:
            msg = f"HTTP {resp.status_code} for {resp.request.method} {resp.request.url}"
        raise TradelogError(msg)

    def _get_json(self, path: str) -> Any:
        resp = self._session.get(self._url(path), timeout=self.timeout, verify=self.verify_ssl)
        if resp.status_code >= 400:
            self._handle_error(resp)
        return resp.json()

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        resp = self._session.post(
            self._url(path),
            json=payload,
            timeout=self.timeout,
            verify=self.verify_ssl,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            self._handle_error(resp)
        return resp.json()

    def _put_json(self, path: str, payload: Mapping[str, Any]) -> None:
        resp = self._session.put(
            self._url(path),
            json=payload,
            timeout=self.timeout,
            verify=self.verify_ssl,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code >= 400:
            self._handle_error(resp)

    def _delete(self, path: str) -> None:
        resp = self._session.delete(self._url(path), timeout=self.timeout, verify=self.verify_ssl)
        if resp.status_code >= 400:
            self._handle_error(resp)
