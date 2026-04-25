from __future__ import annotations

from typing import TYPE_CHECKING

from ib_async import IB, BarData, Contract, Order, Trade

if TYPE_CHECKING:
    from finance.execution.config import EngineConfig
    from finance.execution.strategies.base import OrderSpec


class BrokerError(Exception):
    """Raised when an ib_async operation fails."""


class Broker:
    """Thin async wrapper around ib_async.IB.

    All ib_async exceptions are caught and re-raised as BrokerError so that
    callers never need to handle ib_async internals directly.
    """

    def __init__(self) -> None:
        self._ib = IB()

    async def connect(self, config: EngineConfig) -> None:
        """Connect to IBKR Gateway using the active port from config."""
        try:
            await self._ib.connectAsync(
                host=config.ibkr.host,
                port=config.active_port,
                clientId=config.ibkr.client_id,
            )
        except Exception as exc:
            raise BrokerError(f"Failed to connect: {exc}") from exc

    def disconnect(self) -> None:
        self._ib.disconnect()

    async def qualify_contracts(self, specs: list[Contract]) -> dict[str, Contract]:
        """Qualify a list of Contract specs and return a symbol→Contract map."""
        try:
            qualified = await self._ib.qualifyContractsAsync(*specs)
            return {c.localSymbol: c for c in qualified}
        except Exception as exc:
            raise BrokerError(f"Failed to qualify contracts: {exc}") from exc

    async def fetch_daily_bars(self, contract: Contract, n_days: int) -> list[BarData]:
        """Fetch recent daily bars for ATR calculation."""
        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=f"{n_days} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
            )
            return list(bars)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch daily bars for {contract.localSymbol}: {exc}") from exc

    async def fetch_intraday_bars(self, contract: Contract, bar_size: str) -> list[BarData]:
        """Fetch intraday bars for today's session."""
        try:
            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
            )
            return list(bars)
        except Exception as exc:
            raise BrokerError(
                f"Failed to fetch intraday bars for {contract.localSymbol}: {exc}"
            ) from exc

    async def place_oca_entry(
        self,
        contract: Contract,
        spec: OrderSpec,
        oca_group: str,
    ) -> tuple[Trade, Trade]:
        """Place a BUY STOP + SELL STOP OCA bracket.

        Returns a (long_trade, short_trade) tuple. Both orders share the same
        OCA group so IBKR cancels the unfilled leg automatically on fill.
        """
        try:
            long_order = Order()
            long_order.action = "BUY"
            long_order.orderType = "STP"
            long_order.auxPrice = spec.entry_long
            long_order.totalQuantity = spec.qty
            long_order.ocaGroup = oca_group
            long_order.ocaType = 1
            long_order.transmit = False

            short_order = Order()
            short_order.action = "SELL"
            short_order.orderType = "STP"
            short_order.auxPrice = spec.entry_short
            short_order.totalQuantity = spec.qty
            short_order.ocaGroup = oca_group
            short_order.ocaType = 1
            short_order.transmit = True

            long_trade = self._ib.placeOrder(contract, long_order)
            short_trade = self._ib.placeOrder(contract, short_order)
            return long_trade, short_trade
        except Exception as exc:
            raise BrokerError(f"Failed to place OCA entry for {contract.localSymbol}: {exc}") from exc

    async def place_trailing_stop(
        self,
        contract: Contract,
        direction: str,
        trail_pts: float,
        qty: int,
    ) -> Trade:
        """Place a native IBKR trailing stop after an entry fill."""
        try:
            order = Order()
            order.action = "SELL" if direction == "long" else "BUY"
            order.orderType = "TRAIL"
            order.auxPrice = trail_pts
            order.totalQuantity = qty
            order.transmit = True
            return self._ib.placeOrder(contract, order)
        except Exception as exc:
            raise BrokerError(
                f"Failed to place trailing stop for {contract.localSymbol}: {exc}"
            ) from exc

    async def cancel_order(self, order_id: int) -> None:
        try:
            trades = self._ib.openTrades()
            for trade in trades:
                if trade.order.orderId == order_id:
                    self._ib.cancelOrder(trade.order)
                    return
        except Exception as exc:
            raise BrokerError(f"Failed to cancel order {order_id}: {exc}") from exc

    async def flatten_position(self, contract: Contract) -> None:
        """Cancel all open orders for the contract, then submit a market order to close any open position."""
        try:
            for trade in self._ib.openTrades():
                if trade.contract.localSymbol == contract.localSymbol:
                    self._ib.cancelOrder(trade.order)

            positions = await self._ib.reqPositionsAsync()
            for pos in positions:
                if pos.contract.localSymbol == contract.localSymbol and pos.position != 0:
                    close_order = Order()
                    close_order.action = "SELL" if pos.position > 0 else "BUY"
                    close_order.orderType = "MKT"
                    close_order.totalQuantity = abs(pos.position)
                    close_order.transmit = True
                    self._ib.placeOrder(contract, close_order)
        except Exception as exc:
            raise BrokerError(f"Failed to flatten position for {contract.localSymbol}: {exc}") from exc

    async def get_open_positions(self) -> list:
        try:
            return await self._ib.reqPositionsAsync()
        except Exception as exc:
            raise BrokerError(f"Failed to fetch open positions: {exc}") from exc

    async def get_open_orders(self) -> list:
        try:
            return await self._ib.reqAllOpenOrdersAsync()
        except Exception as exc:
            raise BrokerError(f"Failed to fetch open orders: {exc}") from exc

    @property
    def ib(self) -> IB:
        """Direct access to the underlying IB instance for event subscriptions."""
        return self._ib
