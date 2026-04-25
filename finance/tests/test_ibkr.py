"""
Unit tests for finance.utils.ibkr._build_contract.

All four prefix branches are covered without a live IBKR connection.
ib_async constructors are patched so no network I/O occurs.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import finance.utils.ibkr as ibkr_module
from finance.utils.ibkr import _build_contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_ib(**overrides):
    """Context manager that patches ib_async constructors used by _build_contract."""
    targets = {
        "ib.Forex": "finance.utils.ibkr.ib.Forex",
        "ib.CFD": "finance.utils.ibkr.ib.CFD",
        "ib.Stock": "finance.utils.ibkr.ib.Stock",
    }
    targets.update(overrides)
    return targets


# ---------------------------------------------------------------------------
# 1. Forex branch  (^XXXYYY)
# ---------------------------------------------------------------------------

class TestBuildContractForex:
    def test_eurusd_uses_forex_constructor(self) -> None:
        mock_forex = MagicMock()
        with patch("finance.utils.ibkr.ib.Forex", return_value=mock_forex) as m:
            result = _build_contract("^EURUSD")
        m.assert_called_once_with(symbol="EUR", exchange="IDEALPRO", currency="USD")
        assert result is mock_forex

    def test_gbpusd_uses_forex_constructor(self) -> None:
        with patch("finance.utils.ibkr.ib.Forex") as m:
            _build_contract("^GBPUSD")
        m.assert_called_once_with(symbol="GBP", exchange="IDEALPRO", currency="USD")

    def test_usdjpy_uses_forex_constructor(self) -> None:
        with patch("finance.utils.ibkr.ib.Forex") as m:
            _build_contract("^USDJPY")
        m.assert_called_once_with(symbol="USD", exchange="IDEALPRO", currency="JPY")

    def test_forex_does_not_call_stock(self) -> None:
        with patch("finance.utils.ibkr.ib.Forex"):
            with patch("finance.utils.ibkr.ib.Stock") as stock_mock:
                _build_contract("^EURUSD")
        stock_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 2. CFD branch  ($$NAME)
# ---------------------------------------------------------------------------

class TestBuildContractCFD:
    def test_cfd_strips_double_dollar(self) -> None:
        with patch("finance.utils.ibkr.ib.CFD") as m:
            _build_contract("$$NAS100")
        m.assert_called_once_with(symbol="NAS100", exchange="SMART")

    def test_cfd_does_not_call_stock(self) -> None:
        with patch("finance.utils.ibkr.ib.CFD"):
            with patch("finance.utils.ibkr.ib.Stock") as stock_mock:
                _build_contract("$$NAS100")
        stock_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Index branch  ($NAME)
# ---------------------------------------------------------------------------

class TestBuildContractIndex:
    def test_vix_returns_index_from_indices(self) -> None:
        """$VIX must resolve to the VIX Index object present in INDICES."""
        contract = _build_contract("$VIX")
        # VIX is an ib.Index — check symbol attribute rather than type to avoid
        # importing the ib_async class directly in tests.
        assert contract.symbol == "VIX"

    def test_index_does_not_call_stock(self) -> None:
        with patch("finance.utils.ibkr.ib.Stock") as stock_mock:
            _build_contract("$VIX")
        stock_mock.assert_not_called()

    def test_unknown_index_raises(self) -> None:
        """A $-prefixed symbol not in INDICES should raise IndexError."""
        with pytest.raises(IndexError):
            _build_contract("$UNKNOWN_INDEX_XYZ")


# ---------------------------------------------------------------------------
# 4. Stock branch  (plain symbol)
# ---------------------------------------------------------------------------

class TestBuildContractStock:
    def test_plain_symbol_uses_stock_constructor(self) -> None:
        mock_stock = MagicMock()
        with patch("finance.utils.ibkr.ib.Stock", return_value=mock_stock) as m:
            result = _build_contract("AAPL")
        m.assert_called_once_with(symbol="AAPL", exchange="SMART", currency="USD")
        assert result is mock_stock

    def test_stock_does_not_call_forex(self) -> None:
        with patch("finance.utils.ibkr.ib.Stock"):
            with patch("finance.utils.ibkr.ib.Forex") as forex_mock:
                _build_contract("AAPL")
        forex_mock.assert_not_called()
