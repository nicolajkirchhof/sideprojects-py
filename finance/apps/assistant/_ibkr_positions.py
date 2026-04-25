"""
finance.apps.assistant._ibkr_positions
=========================================
Live position loader from Interactive Brokers TWS (TA-E7-S2).

Connects via ib_async, fetches the portfolio snapshot and open orders,
then maps each position to the Position dataclass defined in _rules.py.

1R computation:
  - Stock: stop_distance × |quantity| where stop_distance is derived from
    the nearest STOP/TRAIL order for the symbol in ib.openOrders().
    Falls back to entry_price × 0.08 × |quantity| if no stop found.
  - Option: |averageCost| × |quantity| × 100  (cost basis = 1R for premium)
"""
from __future__ import annotations

import logging
from datetime import date

log = logging.getLogger(__name__)


def fetch_live_positions(host: str = "127.0.0.1", port: int = 7497) -> list:
    """
    Fetch live portfolio positions from TWS and return a list of Position objects.

    Parameters
    ----------
    host:
        TWS host (default 127.0.0.1).
    port:
        TWS port — 7497 for paper, 7496 for live (default 7497).

    Returns
    -------
    list[Position]
        One Position per open portfolio item; empty list on any error.
    """
    try:
        import ib_async
    except ImportError:
        log.error("ib_async not installed — cannot fetch live positions")
        return []

    try:
        return _fetch_positions(ib_async, host, port)
    except Exception:
        log.exception("fetch_live_positions failed")
        return []


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

_CLIENT_ID = 30   # use a fixed client-id that won't conflict with data tools


def _fetch_positions(ib_async, host: str, port: int) -> list:
    from finance.apps.assistant._rules import Position

    ib = ib_async.IB()
    try:
        ib_async.util.startLoop()
        ib.connect(host, port, clientId=_CLIENT_ID, readonly=True)

        portfolio_items = ib.portfolio()
        open_orders = ib.reqAllOpenOrders()

        stop_prices = _extract_stop_prices(open_orders)
        open_dates = _fetch_open_dates_from_tradelog()

        positions: list[Position] = []
        for item in portfolio_items:
            pos = _map_portfolio_item(item, stop_prices, open_dates)
            if pos is not None:
                positions.append(pos)

        return positions
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def _fetch_open_dates_from_tradelog() -> dict[str, date]:
    """
    Best-effort: fetch open stock positions from Tradelog and return a
    symbol → open_date mapping.  Returns {} if the server is unavailable
    or the response does not contain an openDate field.
    """
    try:
        from finance.apps.assistant._tradelog_client import (
            fetch_open_option_positions,
            fetch_open_stock_positions,
        )

        open_dates: dict[str, date] = {}
        for pos in fetch_open_stock_positions("http://localhost:5186"):
            symbol = pos.get("symbol", "")
            raw_date = pos.get("openDate") or pos.get("open_date") or pos.get("entryDate")
            if symbol and raw_date:
                try:
                    open_dates[symbol] = date.fromisoformat(str(raw_date)[:10])
                except ValueError:
                    pass

        for pos in fetch_open_option_positions("http://localhost:5186"):
            symbol = pos.get("symbol") or pos.get("underlying", "")
            raw_date = pos.get("openDate") or pos.get("open_date") or pos.get("entryDate")
            if symbol and raw_date and symbol not in open_dates:
                try:
                    open_dates[symbol] = date.fromisoformat(str(raw_date)[:10])
                except ValueError:
                    pass

        log.debug("Tradelog open-date enrichment: %d symbols", len(open_dates))
        return open_dates
    except Exception:
        log.debug("Could not fetch open dates from Tradelog", exc_info=True)
        return {}


def _extract_stop_prices(open_orders) -> dict[str, float]:
    """
    Build a symbol → stop_price mapping from open stop/trail orders.

    Only STOP, STOP LIMIT, and TRAIL order types are considered.
    """
    stops: dict[str, float] = {}
    stop_types = {"STP", "STP LMT", "TRAIL", "TRAILLIMIT"}

    for trade in open_orders:
        order = trade.order
        contract = trade.contract
        if not hasattr(order, "orderType"):
            continue
        if order.orderType.upper() not in stop_types:
            continue
        symbol = getattr(contract, "symbol", "")
        if not symbol:
            continue
        # auxPrice is the stop/trail price for STP/TRAIL orders
        price = getattr(order, "auxPrice", None) or getattr(order, "lmtPrice", None)
        if price and price > 0:
            stops[symbol] = float(price)

    return stops


def _map_portfolio_item(
    item,
    stop_prices: dict[str, float],
    open_dates: dict[str, date],
) -> object:
    """Map one ib_async PortfolioItem to a Position, or None to skip."""
    from finance.apps.assistant._rules import Position

    contract = item.contract
    sec_type = getattr(contract, "secType", "").upper()

    if sec_type not in ("STK", "OPT"):
        return None

    symbol: str = getattr(contract, "symbol", "")
    quantity: float = float(item.position)
    avg_cost: float = float(item.averageCost)
    market_price: float = float(item.marketPrice)
    unrealised_pnl: float = float(item.unrealizedPNL)

    if quantity == 0:
        return None

    direction = "long" if quantity > 0 else "short"
    position_type = "stock" if sec_type == "STK" else "option"

    # --- 1R computation ---
    if sec_type == "STK":
        stop = stop_prices.get(symbol)
        if stop is not None:
            stop_distance = abs(avg_cost - stop)
        else:
            # fallback: 8% of entry price
            stop_distance = avg_cost * 0.08
        initial_risk = stop_distance * abs(quantity)
    else:
        # Options: cost basis is 1R; avg_cost is per-share, multiply by 100
        initial_risk = abs(avg_cost) * abs(quantity) * 100

    # Days held — enriched from Tradelog if available; 0 means unknown.
    open_date = open_dates.get(symbol)
    days_held = (date.today() - open_date).days if open_date is not None else 0

    return Position(
        symbol=symbol,
        position_type=position_type,
        direction=direction,
        entry_price=avg_cost,
        current_price=market_price,
        initial_risk=max(initial_risk, 0.01),  # guard against zero
        pnl_dollars=unrealised_pnl,
        days_held=days_held,
    )
