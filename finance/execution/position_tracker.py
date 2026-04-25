from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finance.execution.journal import Journal


@dataclass
class InstrumentState:
    strategy_id: str
    long_order_id: int | None = None
    short_order_id: int | None = None
    direction: str | None = None
    fill_price: float | None = None
    trail_order_id: int | None = None


class PositionTracker:
    """Tracks in-memory state for all instruments being managed by the engine.

    Maintains two indexes:
    - ``_positions``: maps symbol → InstrumentState
    - ``_order_map``: maps order_id → strategy_id for fill routing

    All state transitions are logged to the journal.
    """

    def __init__(self, journal: Journal) -> None:
        self._journal = journal
        self._positions: dict[str, InstrumentState] = {}
        self._order_map: dict[int, str] = {}

    def get(self, symbol: str) -> InstrumentState | None:
        return self._positions.get(symbol)

    def order_id_to_strategy(self, order_id: int) -> str | None:
        return self._order_map.get(order_id)

    def set_pending(
        self,
        *,
        symbol: str,
        strategy_id: str,
        long_order_id: int,
        short_order_id: int,
    ) -> None:
        """Register a newly placed OCA bracket as pending (no fill yet)."""
        state = InstrumentState(
            strategy_id=strategy_id,
            long_order_id=long_order_id,
            short_order_id=short_order_id,
        )
        self._positions[symbol] = state
        self._order_map[long_order_id] = strategy_id
        self._order_map[short_order_id] = strategy_id

    def on_fill(
        self,
        *,
        symbol: str,
        filled_order_id: int,
        direction: str,
        fill_price: float,
        trail_order_id: int | None = None,
    ) -> None:
        """Transition symbol state to filled and record which leg was hit."""
        state = self._positions[symbol]
        state.direction = direction
        state.fill_price = fill_price
        state.trail_order_id = trail_order_id

        if trail_order_id is not None:
            self._order_map[trail_order_id] = state.strategy_id

        self._journal.write({
            "event": "fill",
            "strategy_id": state.strategy_id,
            "symbol": symbol,
            "direction": direction,
            "fill_price": fill_price,
            "filled_order_id": filled_order_id,
            "trail_order_id": trail_order_id,
        })

    def on_close(self, symbol: str) -> None:
        """Remove all state for a symbol after a position is closed."""
        state = self._positions.pop(symbol, None)
        if state is None:
            return
        for oid in (state.long_order_id, state.short_order_id, state.trail_order_id):
            if oid is not None:
                self._order_map.pop(oid, None)

    def reconcile(
        self,
        *,
        ibkr_positions: list,
        ibkr_orders: list,
        strategy_id_for_symbol: dict[str, str],
    ) -> None:
        """Rebuild state from live IBKR data after a reconnect.

        Clears all existing state first, then reconstructs from the provided
        IBKR position and order snapshots. ``strategy_id_for_symbol`` must map
        each active symbol to its owning strategy.
        """
        self._positions.clear()
        self._order_map.clear()

        active_symbols = {p.contract.localSymbol for p in ibkr_positions if p.position != 0}

        for symbol in active_symbols:
            strategy_id = strategy_id_for_symbol.get(symbol, "unknown")
            state = InstrumentState(strategy_id=strategy_id)

            # Determine direction from net position sign
            for pos in ibkr_positions:
                if pos.contract.localSymbol == symbol:
                    state.direction = "long" if pos.position > 0 else "short"

            # Find the trailing stop order for this symbol
            for order in ibkr_orders:
                if (
                    order.contract.localSymbol == symbol
                    and order.order.orderType == "TRAIL"
                ):
                    state.trail_order_id = order.orderId
                    self._order_map[order.orderId] = strategy_id

            self._positions[symbol] = state
