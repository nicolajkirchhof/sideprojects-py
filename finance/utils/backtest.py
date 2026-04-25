"""
finance.utils.backtest
=======================
Shared cost model for all backtests.

Three instrument classes are supported:

  STOCK   — equity trades: flat commission + 0.05% slippage per side
  FUTURE  — index futures: tick spread cost + fixed commission per contract
  OPTION  — options: 20% of estimated bid-ask width per contract

Usage
-----
    from finance.utils.backtest import InstrumentClass, net_pnl, cost_per_trade

    cost = cost_per_trade(InstrumentClass.FUTURE, notional=12.50)
    pnl  = net_pnl(gross, InstrumentClass.STOCK, notional=5_000)
"""
from __future__ import annotations

from enum import Enum

_STOCK_COMMISSION = 1.00       # flat per round trip
_STOCK_SLIPPAGE_PCT = 0.0005   # 0.05% per side = 0.10% round trip
_FUTURE_COMMISSION = 10.00     # per round trip per contract
_OPTION_BAW_FRACTION = 0.20    # fraction of bid-ask width charged per side


class InstrumentClass(Enum):
    STOCK = "stock"
    FUTURE = "future"
    OPTION = "option"


def cost_per_trade(
    instrument_class: InstrumentClass,
    notional: float,
    n_contracts: int = 1,
) -> float:
    """
    Return the total round-trip transaction cost in currency units.

    Parameters
    ----------
    instrument_class:
        Asset class determining the cost model applied.
    notional:
        STOCK  — trade value in dollars (entry price × shares)
        FUTURE — dollar value of one tick spread per contract
        OPTION — estimated bid-ask width in dollars per contract
    n_contracts:
        Number of contracts or share lots (ignored for STOCK where
        cost scales with notional directly).
    """
    if instrument_class is InstrumentClass.STOCK:
        if n_contracts != 1:
            raise ValueError(
                "STOCK cost scales with notional, not contracts. "
                "Pass the full trade value in notional and leave n_contracts=1."
            )
        slippage = notional * _STOCK_SLIPPAGE_PCT * 2
        return _STOCK_COMMISSION + slippage

    if instrument_class is InstrumentClass.FUTURE:
        return (notional + _FUTURE_COMMISSION) * n_contracts

    if instrument_class is InstrumentClass.OPTION:
        return notional * _OPTION_BAW_FRACTION * n_contracts

    raise ValueError(f"Unknown instrument class: {instrument_class}")


def net_pnl(
    gross_pnl: float,
    instrument_class: InstrumentClass,
    notional: float,
    n_contracts: int = 1,
) -> float:
    """Return gross P&L less round-trip transaction costs."""
    return gross_pnl - cost_per_trade(instrument_class, notional, n_contracts)
